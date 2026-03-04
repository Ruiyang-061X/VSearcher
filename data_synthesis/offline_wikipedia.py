import requests
from bs4 import BeautifulSoup
from urllib.parse import urlparse, urljoin, quote


def parse_viewer_url(viewer_url: str):
    """
    Parse a Kiwix viewer URL like:
      http://192.168.72.23:8080/viewer#wikipedia_en_all_maxi_2025-08/Albert_Einstein

    Returns:
        base_url  -> 'http://192.168.72.23:8080'
        zim_id    -> 'wikipedia_en_all_maxi_2025-08'
        article   -> 'Albert_Einstein'
    """
    parsed = urlparse(viewer_url)
    base_url = f"{parsed.scheme}://{parsed.netloc}"

    # fragment is like 'wikipedia_en_all_maxi_2025-08/Albert_Einstein'
    fragment = parsed.fragment.lstrip("/")
    parts = fragment.split("/", 1)
    if len(parts) != 2:
        raise ValueError(f"Unexpected viewer fragment format: {fragment}")

    zim_id, article = parts
    return base_url, zim_id, article


def build_article_url(base_url: str, zim_id: str, article_title: str) -> str:
    """
    Build the direct article URL on kiwix-serve.

    This pattern:
        http://<host>/<zim_id>/A/<title>

    is the common one for kiwix-serve; if your instance uses a different
    pattern, adjust this function accordingly.
    """
    safe_title = quote(article_title)
    return f"{base_url.rstrip('/')}/{zim_id}/A/{safe_title}"


def extract_text_and_image(html: str, page_url: str):
    """
    Given article HTML, return (text_content, image_url).

    - Text: concatenation of all non-empty <p> in the main content.
    - Image: first image in infobox if present, otherwise first <img>.
    """
    soup = BeautifulSoup(html, "html.parser")

    # Try to find the main content container
    content = (
        soup.find("div", id="content")
        or soup.find("div", id="mw-content-text")
        or soup.body
    )

    if content is None:
        return "", None

    # Collect paragraphs
    paragraphs = []
    for p in content.find_all("p", recursive=True):
        text = p.get_text(strip=True)
        if text:
            paragraphs.append(text)

    text_content = "\n\n".join(paragraphs)

    # Try to find a representative image
    img_tag = None

    # Prefer infobox image (typical Wikipedia layout)
    infobox = soup.find("table", class_="infobox")
    if infobox:
        img_tag = infobox.find("img")

    # Fallback: first image anywhere
    if not img_tag:
        img_tag = content.find("img")

    image_url = None
    if img_tag and img_tag.get("src"):
        # Kiwix usually uses relative paths for images, so make it absolute
        image_url = urljoin(page_url, img_tag["src"])

    return text_content, image_url


def get_entity_from_viewer_url(viewer_url: str):
    """
    High-level helper:
    - Takes a viewer URL
    - Resolves it to the direct article URL
    - Fetches and parses text + image
    """
    base_url, zim_id, article = parse_viewer_url(viewer_url)
    article_url = build_article_url(base_url, zim_id, article)

    resp = requests.get(article_url)
    resp.raise_for_status()

    text, image_url = extract_text_and_image(resp.text, article_url)

    return {
        "title": article.replace("_", " "),
        "article_url": article_url,
        "text": text,
        "image_url": image_url,
    }


def obtain_entity_information(entity_name: str, base_url="http://192.168.72.23:8080", zim_id="wikipedia_en_all_maxi_2025-08"):
    article = entity_name.replace(" ", "_")
    article_url = build_article_url(base_url, zim_id, article)

    resp = requests.get(article_url)
    resp.raise_for_status()

    text, image_url = extract_text_and_image(resp.text, article_url)

    return {
        "title": entity_name,
        "article_url": article_url,
        "text": text,
        "image_url": image_url,
    }


if __name__ == "__main__":
    entity = "Albert Einstein"
    result = obtain_entity_information(entity)

    print(result["title"])
    print(result["article_url"])
    print(result["text"][:500])
    print(result["image_url"])