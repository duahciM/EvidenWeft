from pathlib import Path
from if10_research import freeze

freeze.HERE = Path(__file__).resolve().parent
freeze.CARD = freeze.MEDIA.parent / "video-keyframes-lightweight-master-IF14-identity-replacement-next.md"

if __name__ == "__main__":
    freeze.main()
