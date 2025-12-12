import os
import sys
from urllib.parse import urlparse

from dotenv import load_dotenv
from mlx_audio.server import main

load_dotenv()

if __name__ == "__main__":
    # Add the port from the API base URL to the arguments to start the server at the defined prot.
    base_url = os.getenv("API_BASE_URL_MLX_AUDIO")
    if base_url is not None:
        parsed = urlparse(base_url)
        sys.argv.extend(["--port", str(parsed.port)])

    main()
