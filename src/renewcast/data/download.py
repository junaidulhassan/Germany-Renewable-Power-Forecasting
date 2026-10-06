"""Download DWD HOSTRADA hourly NetCDF files (one file per month and variable)."""

from __future__ import annotations

import logging
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.parse import urljoin

import requests

from renewcast.config import Settings

logger = logging.getLogger(__name__)

_HREF_RE = re.compile(r'href="([^"]+\.nc)"')


def list_remote_files(session: requests.Session, url: str, timeout: int) -> list[str]:
    """Return the ``.nc`` file names listed on a DWD directory index page."""
    response = session.get(url, timeout=timeout)
    response.raise_for_status()
    return sorted(set(_HREF_RE.findall(response.text)))


def select_files(names: list[str], variable: str, file_stem: str, years: list[int]) -> list[str]:
    """Keep monthly files of ``variable`` for ``years``.

    Example name: ``sfcWind_1hr_HOSTRADA-v1-0_BE_gn_2023010100-2023013123.nc``
    """
    year_alt = "|".join(str(y) for y in years)
    pattern = re.compile(
        rf"^{re.escape(variable)}_1hr_{re.escape(file_stem)}_(?:{year_alt})\d{{2}}0100-\d+\.nc$"
    )
    return [n for n in names if pattern.match(n)]


def download_file(
    session: requests.Session, url: str, dest: Path, timeout: int, max_retries: int
) -> Path:
    """Stream ``url`` to ``dest`` via a ``.part`` file so interrupted downloads never look complete."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_suffix(dest.suffix + ".part")
    last_error: Exception | None = None
    for attempt in range(1, max_retries + 1):
        try:
            with session.get(url, stream=True, timeout=timeout) as response:
                response.raise_for_status()
                with open(part, "wb") as fh:
                    for chunk in response.iter_content(chunk_size=1 << 20):
                        fh.write(chunk)
            part.replace(dest)
            return dest
        except (requests.RequestException, OSError) as exc:
            last_error = exc
            wait = 2**attempt
            logger.warning(
                "Download failed (%s), attempt %d/%d, retrying in %ds",
                exc,
                attempt,
                max_retries,
                wait,
            )
            time.sleep(wait)
    raise RuntimeError(f"Giving up on {url}: {last_error}")


def download_weather(
    settings: Settings,
    variables: list[str] | None = None,
    dry_run: bool = False,
) -> dict[str, list[Path]]:
    """Download all monthly files for the configured years. Returns local paths per variable."""
    variables = variables or settings.download.variables
    cfg = settings.download
    result: dict[str, list[Path]] = {}
    with requests.Session() as session:
        for variable in variables:
            spec = settings.weather.variables[variable]
            base = f"{settings.weather.base_url.rstrip('/')}/{spec.directory}/"
            names = list_remote_files(session, base, cfg.timeout_s)
            wanted = select_files(names, variable, settings.weather.file_stem, settings.years)
            if not wanted:
                raise RuntimeError(
                    f"No files for '{variable}' {settings.years} found at {base}. "
                    "Check weather.variables[...].directory in the config against the DWD index."
                )
            out_dir = settings.paths.weather_raw_dir / variable
            logger.info("%s: %d files selected from %s", variable, len(wanted), base)
            paths = [out_dir / n for n in wanted]
            result[variable] = paths
            if dry_run:
                for n in wanted:
                    logger.info("[dry-run] would download %s", urljoin(base, n))
                continue
            todo = [
                (n, p)
                for n, p in zip(wanted, paths, strict=True)
                if not (cfg.skip_existing and p.exists())
            ]
            logger.info(
                "%s: %d to download, %d already present",
                variable,
                len(todo),
                len(wanted) - len(todo),
            )
            with ThreadPoolExecutor(max_workers=cfg.workers) as pool:
                futures = {
                    pool.submit(
                        download_file, session, urljoin(base, n), p, cfg.timeout_s, cfg.max_retries
                    ): n
                    for n, p in todo
                }
                for future in as_completed(futures):
                    future.result()
                    logger.info("Downloaded %s", futures[future])
    return result
