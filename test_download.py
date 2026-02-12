from src.skills.media_fetcher import MediaFetcher

fetcher = MediaFetcher()
# Prova con un URL di un reel pubblico
result = fetcher.download("https://www.instagram.com/reel/DLdc7-KM3ux/")
print(f"File salvato in: {result}")