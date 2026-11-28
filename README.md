# EvidenWeft

Core offline video-evidence algorithms. The source contains D1–D6 nominations, Figure 1/2 measurement, discrete timeline cues, evidence selection, and an MCP service for prepared packets.

Install `requirements.txt`, then run:

```sh
python -m p224_research.mcp_server --packet <prepared.json> --log <calls.jsonl> --wave
```

The MCP service accepts an existing packet and checks original PNG hashes and dimensions. The service returns navigation metadata or exact originals at registered timestamps. Raw-video ingest for end users is available in [VisuReach](https://github.com/duahciM/dsh-visureach).

[Research records and protocols](https://github.com/duahciM/EvidenWeft-Archive) · [Historical source releases](https://github.com/duahciM/EvidenWeft/releases)
