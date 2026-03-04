from google.cloud import vision
import requests
from urllib.parse import urlparse

from inference.inference_util import *


class ImageSearch:

    name = "image_search"

    def __init__(self):
        self.client = vision.ImageAnnotatorClient()
        self.top_k = 5
        self.cache = load_json(".cache/image_search.json")

    def call(self, params):
        if params["image_url"] in self.cache:
            search_results = self.cache[params["image_url"]]['search_results']
            search_results_image = self.cache[params["image_url"]]['search_results_image']
        else:
            image = vision.Image()
            image.source.image_uri = params["image_url"]

            response = self.client.web_detection(image=image)
            annotations = response.web_detection
            cnt = 0
            search_results = ""
            search_results_image = []
            for page in annotations.pages_with_matching_images:
                if cnt >= self.top_k:
                    break
                page_image = self.obtain_page_image(page)
                if page_image:
                    search_results += f"{cnt + 1}. [Page Link] {page.url} [Page Title] {page.page_title}\n"
                    search_results_image.append(page_image)
                    cnt += 1
            self.cache[params["image_url"]] = {
                "search_results": search_results,
                "search_results_image": search_results_image
            }
            if len(self.cache) % 10 == 0:
                save_json(self.cache, ".cache/image_search.json")
        return search_results, search_results_image

    def obtain_page_image(self, page):
        if not page.full_matching_images and not page.partial_matching_images:
            return None
        
        for image in page.full_matching_images:
            if self.check_image_url_valid(image.url):
                return image.url

        for image in page.partial_matching_images:
            if self.check_image_url_valid(image.url):
                return image.url

    def check_image_url_valid(self, image_url):
        if not image_url:
            return False
        parsed = urlparse(image_url)
        if not (parsed.scheme and parsed.netloc):
            return False
        try:
            resp = requests.head(image_url, timeout=5, allow_redirects=True)
            if resp.status_code >= 400 or "content-type" not in resp.headers:
                resp = requests.get(image_url, stream=True, timeout=5)
            if resp.status_code >= 400:
                return False
            content_type = resp.headers.get("content-type", "").lower()
            if not content_type.startswith("image/"):
                return False
            return True
        except requests.exceptions.RequestException:
            return False

if __name__ == "__main__":
    image_search = ImageSearch()
    params = {
        "image_url": "https://raw.githubusercontent.com/image-storage-rl/image_storage/main/mmsearch/mmsearch_276.jpg"
    }
    search_results, search_results_image = image_search.call(params)
    print(search_results)
    print(search_results_image)