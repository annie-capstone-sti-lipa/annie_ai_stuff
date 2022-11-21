import requests
import json
import re

data_set = []


def clean(text):
    return re.sub('[^A-Za-z0-9 ]+', "", remove_special(text)).lower()


def remove_special(text):
    return text.encode("ascii", "ignore").decode().replace("*", "")


def get_anime_list(offset=0, count=0):
    response = requests.get(
        "https://api.myanimelist.net/v2/anime/ranking",
        params={
            "limit": 100,
            "ranking_type": "all",
            "fields": "alternative_titles",
            "offset": offset,
        },
        headers={"X-MAL-CLIENT-ID": "64f2631feddc2a7774f45e2cc49cd7f4"}
    )

    animes = response.json()

    for anime in animes["data"]:
        node = anime["node"]

        alternatives = node["alternative_titles"]
        alternative = []

        # alternative.extend([clean(alt)
        #                    for alt in alternatives["synonyms"]])
        alternative.append(clean(alternatives["en"]))
        alternative.append(clean(node["title"]))
        alternative.append(clean(node["title"]))
        alternative.append(clean(node["title"]))

        print(node["title"])
        print(alternatives["en"])

        alternative = list(filter(None, [a for a in alternative if a.strip()]))

        temp_data = {"title": clean(
            node["title"]), "alternative": alternative}

        count += len(temp_data["alternative"])
        count += 1

        data_set.append(temp_data)

    return count


def write_data():
    with open("anime_titles.json", "w", encoding="utf8") as outfile:
        json.dump(data_set, outfile)

    print("\n\n\nDONEZO!\n\n\n")


count = 0
for _ in range(25):
    count = get_anime_list(offset=len(data_set), count=count)

print(count)
write_data()
