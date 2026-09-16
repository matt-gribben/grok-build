use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use axum::Router;
use axum::body::Body;
use axum::http::{Request, Response};
use axum::routing::post;
use bytes::Bytes;
use futures_util::StreamExt;
use prost::Message;
use rcgen::{CertificateParams, KeyPair};
use rustls::pki_types::PrivateKeyDer;

use crate::cursor_connect::{ConnectFrameDecoder, encode_connect_frame};
use crate::cursor_proto::agent::v1::{
    AgentClientMessage, AgentServerMessage, InteractionUpdate, TextDeltaUpdate, TurnEndedUpdate,
    agent_client_message, agent_server_message, interaction_update,
};
use crate::cursor_transport::{TestCursorTransportGuard, install_test_transport};

const RUN_RPC: &str = "/agent.v1.AgentService/Run";

pub struct FakeCursorServer {
    run_requests: Arc<AtomicUsize>,
    server: tokio::task::JoinHandle<()>,
    _transport: TestCursorTransportGuard,
}

impl FakeCursorServer {
    pub async fn start(session_id: impl Into<String>, response_text: impl Into<String>) -> Self {
        let _ = rustls::crypto::aws_lc_rs::default_provider().install_default();
        let response_text = Arc::new(response_text.into());
        let run_requests = Arc::new(AtomicUsize::new(0));
        let key_pair = KeyPair::generate().expect("test key");
        let certificate = CertificateParams::new(vec!["localhost".to_owned()])
            .expect("certificate params")
            .self_signed(&key_pair)
            .expect("self-sign");
        let private_key = PrivateKeyDer::Pkcs8(key_pair.serialize_der().into());
        let mut config = rustls::ServerConfig::builder()
            .with_no_client_auth()
            .with_single_cert(vec![certificate.der().clone()], private_key)
            .expect("server config");
        config.alpn_protocols = vec![b"h2".to_vec()];
        let tls = axum_server::tls_rustls::RustlsConfig::from_config(Arc::new(config));
        let app = Router::new().route(
            RUN_RPC,
            post({
                let run_requests = Arc::clone(&run_requests);
                let response_text = Arc::clone(&response_text);
                move |request| run_endpoint(request, run_requests, response_text)
            }),
        );
        let listener = std::net::TcpListener::bind("127.0.0.1:0").expect("bind listener");
        listener
            .set_nonblocking(true)
            .expect("nonblocking listener");
        let port = listener.local_addr().expect("listener address").port();
        let server = tokio::spawn(async move {
            axum_server::from_tcp_rustls(listener, tls)
                .expect("build TLS server")
                .serve(app.into_make_service())
                .await
                .expect("serve fake Cursor");
        });
        let certificate =
            reqwest::Certificate::from_der(certificate.der().as_ref()).expect("test certificate");
        let transport = install_test_transport(
            session_id,
            &format!("https://localhost:{port}"),
            certificate,
            "fixture-cursor-token",
        );
        Self {
            run_requests,
            server,
            _transport: transport,
        }
    }

    pub fn run_requests(&self) -> usize {
        self.run_requests.load(Ordering::Acquire)
    }
}

impl Drop for FakeCursorServer {
    fn drop(&mut self) {
        self.server.abort();
    }
}

async fn run_endpoint(
    request: Request<Body>,
    run_requests: Arc<AtomicUsize>,
    response_text: Arc<String>,
) -> Response<Body> {
    let mut body = request.into_body().into_data_stream();
    let mut decoder = ConnectFrameDecoder::default();
    let run = loop {
        let bytes = body
            .next()
            .await
            .expect("initial request frame")
            .expect("initial request bytes");
        let frames = decoder.push(&bytes).expect("initial request Connect frame");
        if let Some(frame) = frames.first() {
            break AgentClientMessage::decode(frame.payload.as_slice())
                .expect("initial Run request");
        }
    };
    assert!(matches!(
        run.message,
        Some(agent_client_message::Message::RunRequest(_))
    ));
    run_requests.fetch_add(1, Ordering::AcqRel);

    let text = AgentServerMessage {
        message: Some(agent_server_message::Message::InteractionUpdate(
            InteractionUpdate {
                message: Some(interaction_update::Message::TextDelta(TextDeltaUpdate {
                    text: response_text.as_ref().clone(),
                })),
                ..Default::default()
            },
        )),
        ..Default::default()
    };
    let turn_ended = AgentServerMessage {
        message: Some(agent_server_message::Message::InteractionUpdate(
            InteractionUpdate {
                message: Some(interaction_update::Message::TurnEnded(TurnEndedUpdate {})),
                ..Default::default()
            },
        )),
        ..Default::default()
    };
    let response_stream = async_stream::stream! {
        yield Ok::<Bytes, std::convert::Infallible>(
            Bytes::from(encode_connect_frame(0, &text.encode_to_vec()).expect("text frame"))
        );
        yield Ok(Bytes::from(
            encode_connect_frame(0, &turn_ended.encode_to_vec()).expect("turn-ended frame")
        ));
    };
    Response::builder()
        .header("content-type", "application/connect+proto")
        .body(Body::from_stream(response_stream))
        .expect("Connect response")
}
