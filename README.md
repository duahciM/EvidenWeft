# EvidenWeft

Standalone video-evidence algorithms. The current source is IF-30 (P2-24).

The entry points are in `p224_research/`. Install the Python dependencies in
`requirements.txt` and run this from the repository root.

```sh
python -m p224_research.mcp_server --help
```

Some shared modules retain the name of the research stage that introduced
them. The MCP server needs a prepared video packet. Experiment runners may
also need original media, local model assets and the frozen experiment inputs.

[Download the current source](https://github.com/duahciM/EvidenWeft/releases/latest) or browse [historical releases](https://github.com/duahciM/EvidenWeft/releases).
