from src.skills.media_fetcher import MediaFetcher
import json

def test_fetcher():
    mf = MediaFetcher(temp_dir="temp/test_media")
    urls = {
        "reel": "https://www.instagram.com/reel/DUY8rBwEgvT/?utm_source=ig_web_copy_link&igsh=NTc4MTIwNjQ2YQ==",
        "carousel": "https://www.instagram.com/p/DQ4HibODu5C/?img_index=1",
        "image": "https://www.instagram.com/p/DUn6Iz6DSiO/?utm_source=ig_web_copy_link&igsh=NTc4MTIwNjQ2YQ=="
    }

    for label, url in urls.items():
        print(f"\n--- Testing {label} ---")
        try:
            result = mf.fetch(url)
            print(json.dumps(result, indent=2))
        except Exception as e:
            print(f"Errore su {label}: {e}")

if __name__ == "__main__":
    test_fetcher()