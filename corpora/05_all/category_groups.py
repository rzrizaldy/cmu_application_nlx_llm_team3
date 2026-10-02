"""Shared subtopic split from the team brief (brief/team_corpus_brief.pdf)."""

CATEGORY_GROUPS = {
    "streets_mobility": [
        "Road/Street Issues",
        "Traffic and Street Sign Issues",
        "Street Light",
        "Pedestrian/Bicycle Concerns",
        "Parking",
    ],
    "waste_neighborhood": [
        "Garbage and Litter Issues",
        "Neighborhood Issues",
        "Graffiti Issues",
        "Weeds/Debris",
    ],
    "buildings_construction": [
        "Building Maintenance",
        "Construction Issues",
        "Permits",
        "Accessibility",
    ],
    "parks_public_spaces": [
        "Parks Issues",
        "Tree Issues",
        "Animal Issues",
        "City Facilities and Infrastructure",
    ],
}

# Lead order matches the brief's subtopic table. The folder name is shared by
# corpora/, sources/*.zip and memo/*.pdf. corpora/05_all is the merge output.
SUBTOPICS = [
    {"lead": 1, "key": "streets_mobility", "member": "afaq",
     "title": "Streets and Mobility"},
    {"lead": 2, "key": "waste_neighborhood", "member": "rutomo",
     "title": "Waste and Neighborhood Cleanliness"},
    {"lead": 3, "key": "buildings_construction", "member": "mahika",
     "title": "Buildings, Construction, and Accessibility"},
    {"lead": 4, "key": "parks_public_spaces", "member": "mingchin",
     "title": "Parks, Trees, Animals, and Public Facilities"},
]

for _s in SUBTOPICS:
    _s["folder"] = f"{_s['lead']:02d}_{_s['key']}_{_s['member']}"
