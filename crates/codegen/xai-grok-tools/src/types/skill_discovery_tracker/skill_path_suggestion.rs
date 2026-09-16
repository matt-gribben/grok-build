//! Registered skill-path lookup for failed reads under a skill's directory.

use std::collections::HashSet;
use std::path::{Path, PathBuf};

use crate::implementations::skills::types::skill_name_from_path;

use super::{SkillManager, canonical_path};

/// A unique registered skill file matching a failed read under a skill's directory.
#[derive(Debug, Clone)]
pub(crate) struct SkillPathSuggestion {
    /// Real path used by the filesystem backend.
    pub(crate) path: PathBuf,
    /// Model-facing path with a forked worktree prefix rewritten when needed.
    pub(crate) display_path: PathBuf,
}

/// One deduplicated, enabled skill registration eligible for ancestor matching.
struct Candidate<'a> {
    /// The skill's own directory name (parent of its `SKILL.md`).
    directory_name: &'a str,
    /// Declared skill name, which may differ from `directory_name`.
    name: &'a str,
    /// Root directory containing `SKILL.md` and any companion files (personas, scripts, docs).
    root: &'a Path,
}

impl SkillManager {
    /// Find one registered skill whose directory a failed read's path passes through. Handles a
    /// wrong-root guess for `SKILL.md` itself *and* for any companion file inside the skill's own
    /// directory tree (e.g. `personas/reviewer.md`), since the model can mis-root either kind of
    /// read the same way. Walks `requested_path`'s ancestors from the immediate parent outward and
    /// stops at the first ancestor whose name matches a registered skill's directory name or
    /// declared name; an ambiguous match at that ancestor returns `None` rather than guessing. A
    /// shallower, coincidental ancestor segment (e.g. a generic `skills` directory) never matches
    /// a real skill name, so it is safely skipped in favor of the actual skill directory further
    /// down. Candidates come from the current collections in precedence order — the listing
    /// baseline, held conditional skills, then dynamic discoveries — so a baseline reload that
    /// removes, moves, or disables a skill immediately stops suggesting it.
    pub(crate) fn suggest_skill_path(&self, requested_path: &Path) -> Option<SkillPathSuggestion> {
        // Fork/display state must be coherent before any path is surfaced:
        // half-seeded state could leak a real worktree path to the model.
        let display_mapping = match (&self.real_cwd_prefix, &self.display_cwd) {
            (Some(real), Some(display)) => Some((real.as_str(), display.as_str())),
            (None, None) => None,
            _ => return None,
        };

        let mut owned_paths = HashSet::new();
        let mut candidates: Vec<Candidate<'_>> = Vec::new();
        for skill in self
            .startup_skills
            .iter()
            .chain(self.conditional.held())
            .chain(&self.discovered_skills)
        {
            let skill_path = Path::new(&skill.path);
            if !skill_path.is_absolute() {
                continue;
            }
            let canonical = canonical_path(&skill.path);
            // The highest-precedence record owns its canonical path outright:
            // a shadowed record must not be suggested (nor count as ambiguity)
            // even when the owner is disabled or otherwise ineligible.
            if !owned_paths.insert(canonical) {
                continue;
            }
            if !skill.enabled {
                continue;
            }
            let Some(directory_name) = skill_name_from_path(&skill.path) else {
                continue;
            };
            let Some(root) = skill_path.parent() else {
                continue;
            };
            candidates.push(Candidate {
                directory_name,
                name: &skill.name,
                root,
            });
        }

        for ancestor in requested_path.parent()?.ancestors() {
            let Some(ancestor_name) = ancestor.file_name().and_then(|n| n.to_str()) else {
                continue;
            };
            // Count every eligible same-name registration at this ancestor level: two skills
            // sharing a directory or declared name make the guess ambiguous.
            let matches: Vec<&Candidate<'_>> = candidates
                .iter()
                .filter(|c| c.directory_name == ancestor_name || c.name == ancestor_name)
                .collect();
            if matches.is_empty() {
                continue;
            }
            if matches.len() > 1 {
                return None;
            }
            let candidate = matches[0];
            let relative = requested_path.strip_prefix(ancestor).ok()?;
            let real_path = candidate.root.join(relative);
            if real_path == requested_path {
                // Already the failed read target — not a suggestion.
                return None;
            }
            let display_path = match display_mapping {
                Some((real, display)) => real_path
                    .strip_prefix(real)
                    .map(|rel| Path::new(display).join(rel))
                    .unwrap_or_else(|_| real_path.clone()),
                None => real_path.clone(),
            };
            return Some(SkillPathSuggestion {
                path: real_path,
                display_path,
            });
        }
        None
    }
}

#[cfg(test)]
#[path = "skill_path_suggestion_tests.rs"]
mod tests;
