"""Exact Pittsburgh 311 category filters from the group brief (shared dictionary)."""

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

MEMBER_SUBTOPIC = {
    "afaq": "streets_mobility",
    "rutomo": "waste_neighborhood",
    "mahika": "buildings_construction",
    "mingchin": "parks_public_spaces",
}

MEMBER_DISPLAY = {
    "afaq": "Streets and Mobility (Afaq Khan)",
    "rutomo": "Waste and Neighborhood Cleanliness (Rutomo)",
    "mahika": "Buildings, Construction, and Accessibility (Mahika Gunjkar)",
    "mingchin": "Parks, Trees, Animals, and Public Facilities (Mingchin)",
}
