# AI agent prompt samples
This folder contains samples of prompts that can be used by Generative AI agents
to enhance the Ripple programming experience.

- ripple-hexagon-assistant.md leverages the optimization guides
  to help developers detect potential performance issues in their code.

## Claude Code skill

`ripple-assistant.SKILL.md` is a [Claude Code skill](https://code.claude.com/docs/en/skills.md)
version of the assistant above.

### Installation

Claude Code only discovers skills from a directory named after the skill, so
copy (or symlink) the file into one of these locations and rename it to
`SKILL.md`:

- **Personal** (available in every project on this machine)

  ```bash
  mkdir -p ~/.claude/skills/ripple-optimize
  cp ripple-assistant.SKILL.md \
     ~/.claude/skills/ripple-optimize/SKILL.md
  ```

- **Project** (checked into the repo, shared with collaborators)

  ```bash
  mkdir -p <your-project>/.claude/skills/ripple-optimize
  cp ripple-assistant.SKILL.md \
     <your-project>/.claude/skills/ripple-optimize/SKILL.md
  ```

The target directory name (`ripple-optimize`) must match the `name:`
field in the skill's frontmatter.

### How to trigger it

Once installed, the skill can be activated two ways:

- **Explicitly**, by typing `/ripple-optimize` at the Claude Code
  prompt (optionally followed by a file path or a question).
- **Automatically**, whenever your conversation matches the skill's
  `when_to_use` criteria — for example asking Claude to "review this ripple
  kernel", "optimize this kernel in `foo.cpp`", or "look at the -Rpass output
  for this file".