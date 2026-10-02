import pandas as pd
from example_extraction import make_record
from schema import validate_record

requests = pd.read_csv(
    "311_data.csv",
    dtype={"request_type_id": "string"}
)

codebook = pd.read_csv(
    "311_issue_category_codebook.csv",
    dtype={"request_type_id": "string"}
)

print("311 data rows:", len(requests))
print("Codebook rows:", len(codebook))

print("\n311 data columns:")
print(requests.columns.tolist())

print("\nCodebook columns:")
print(codebook.columns.tolist())

# Clean the codebook
codebook_clean = (
    codebook.loc[
        codebook["request_type_id"].notna()
        & codebook["category"].fillna("").ne("")
    ]
    # Keep the latest version
    .sort_values("codebook_version")
    .drop_duplicates("request_type_id", keep="last")
)

print("\nClean codebook rows:", len(codebook_clean))
print(
    codebook_clean[
        ["request_type_id", "issue", "category", "department"]
    ].head()
)

# Join the request data with the cleaned codebook.
requests_enriched = requests.merge(
    codebook_clean[
        ["request_type_id", "issue", "category", "department"]
    ],
    on="request_type_id",
    how="left",
    validate="many_to_one"
)

print("\nJoined rows:", len(requests_enriched))
print(
    requests_enriched[
        ["request_type_id", "request_type_name", "issue", "category", "department"]
    ].head()
)

# Keep only my categories.
my_categories = [
    "Parks Issues",
    "Tree Issues",
    "Animal Issues",
    "City Facilities and Infrastructure"
]

parks_data = requests_enriched[
    requests_enriched["category"].isin(my_categories)
].copy()

print("\nMy subtopic rows:", len(parks_data))
print(parks_data["category"].value_counts())

# Check the issues inside each category.
for category in my_categories:
    print(f"\nCategory: {category}")
    print(
        parks_data.loc[
            parks_data["category"] == category,
            "issue"
        ].value_counts()
    )

# Check issue definitions for my subtopic.
my_codebook = codebook_clean[
    codebook_clean["category"].isin(my_categories)
]

print("\nIssue definitions:")
print(
    my_codebook[
        ["issue", "category", "department", "definition"]
    ].to_string(index=False)
)

# Preview request records in my subtopic.
print("\nSample request records:")
print(
    parks_data[
        [
            "request_type_name",
            "status_name",
            "dept",
            "origin",
            "neighborhood",
            "create_date_et",
            "closed_date_et",
            "issue",
            "category",
            "department"
        ]
    ].head(10).to_string(index=False)
)

# Check issue IDs for my subtopic.
print("\nIssue reference:")
print(
    my_codebook[
        ["request_type_id", "issue", "category", "department"]
    ]
    .sort_values(["category", "issue"])
    .to_string(index=False)
)

# Corpus design.
RECORDS_PER_ISSUE = 8
TEXT_RECORDS_PER_ISSUE = 6

# Complaint styles for text records.
COMPLAINT_STYLES = [
    "clear",
    "ambiguous",
    "incomplete",
    "typo_heavy",
    "informal",
    "short"
]

# Test complaint examples for one issue.
pruning_examples = {
    "clear": "A city tree in front of my property has several large branches that need pruning.",
    "ambiguous": "The tree by the street is getting too big and needs attention.",
    "incomplete": "There is a tree problem and some branches are hanging down.",
    "typo_heavy": "city tree branchs need prunning they are getting too long",
    "informal": "The street tree is getting kinda out of control. Can someone trim it?",
    "short": "A tree near the sidewalk has several branches that need trimming."
}

# Convert the complaint examples into corpus records.
pruning_records = []

for i, (style, text) in enumerate(pruning_examples.items(), start=1):
    record = make_record(
        doc_id=f"parks_pruning_{i:03d}",
        source_url="https://data.wprdc.org/dataset/311-data/resource/6a2c9de6-9cb8-4da6-bd78-0a913cc5790c",
        license_note="Creative Commons Attribution; subject to WPRDC terms of use.",
        raw_text=text,
        metadata={
            "group_topic": "Pittsburgh 311 Municipal Service Requests",
            "subtopic": "Parks, Trees, Animals, and Public Facilities",
            "request_type_id": "520",
            "issue": "Pruning (city tree)",
            "category": "Tree Issues",
            "department": "DPW - Forestry Division",
            "example_style": style,
            "synthetic": True
        }
    )

    pruning_records.append(record)

print("\nPruning text records:")
for record in pruning_records:
    print(record["doc_id"], "-", record["metadata"]["example_style"], "-", record["raw_text"])

# Validate the pruning records.
print("\nPruning record validation:")

for record in pruning_records:
    problems = validate_record(record)
    print(record["doc_id"], problems)

# Convert dates to datetime.
parks_data["create_date_et"] = pd.to_datetime(
    parks_data["create_date_et"],
    errors="coerce"
)

parks_data["closed_date_et"] = pd.to_datetime(
    parks_data["closed_date_et"],
    errors="coerce"
)

# Calculate resolution time in hours.
parks_data["resolution_hours"] = (
    parks_data["closed_date_et"] - parks_data["create_date_et"]
).dt.total_seconds() / 3600

# Keep valid closed requests.
closed_requests = parks_data[
    parks_data["resolution_hours"].notna()
    & (parks_data["resolution_hours"] >= 0)
].copy()

# Count all requests for each issue.
request_counts = (
    parks_data
    .groupby(["request_type_id", "issue", "category", "department"])
    .size()
    .reset_index(name="request_volume")
)

# Calculate resolution statistics using closed requests only.
resolution_stats = (
    closed_requests
    .groupby(["request_type_id", "issue", "category", "department"])
    .agg(
        median_hours=("resolution_hours", "median"),
        p75_hours=("resolution_hours", lambda x: x.quantile(0.75)),
        p90_hours=("resolution_hours", lambda x: x.quantile(0.90))
    )
    .reset_index()
)

# Combine volume and resolution statistics.
issue_stats = request_counts.merge(
    resolution_stats,
    on=["request_type_id", "issue", "category", "department"],
    how="left"
)

print("\nIssue resolution statistics:")
print(issue_stats.to_string(index=False))

# Convert resolution time from hours to days.
issue_stats["median_days"] = issue_stats["median_hours"] / 24
issue_stats["p75_days"] = issue_stats["p75_hours"] / 24
issue_stats["p90_days"] = issue_stats["p90_hours"] / 24

# Preview the statistics in days.
print("\nIssue statistics in days:")
print(
    issue_stats[
        [
            "issue",
            "category",
            "request_volume",
            "median_days",
            "p75_days",
            "p90_days"
        ]
    ].round(2).to_string(index=False)
)

# Save issue statistics.
issue_stats.to_csv(
    "parks_issue_statistics.csv",
    index=False
)

# Create one test corpus record.
test_record = make_record(
    doc_id="parks_test_001",
    source_url="https://data.wprdc.org/dataset/311-data/resource/6a2c9de6-9cb8-4da6-bd78-0a913cc5790c",
    license_note="Creative Commons Attribution; subject to WPRDC terms of use.",
    raw_text=(
        "A resident reports an issue related to a city tree that needs pruning."
    ),
    metadata={
        "group_topic": "Pittsburgh 311 Municipal Service Requests",
        "subtopic": "Parks, Trees, Animals, and Public Facilities",
        "request_type_id": "520",
        "issue": "Pruning (city tree)",
        "category": "Tree Issues",
        "department": "DPW - Forestry Division"
    }
)

print("\nTest corpus record:")
print(test_record)

# Validate the test record.
problems = validate_record(test_record)

print("\nValidation problems:")
print(problems)

# Get statistics for Pruning (city tree).
pruning_stats = issue_stats[
    issue_stats["issue"] == "Pruning (city tree)"
].iloc[0]

print("\nPruning statistics:")
print(pruning_stats)

# Create a table record for request volume.
pruning_volume_record = make_record(
    doc_id="parks_pruning_007",
    source_url="https://data.wprdc.org/dataset/311-data/resource/29462525-62a6-45bf-9b5e-ad2e1c06348d",
    license_note="Creative Commons Attribution; subject to WPRDC terms of use.",
    table_json={
        "request_type_id": "520",
        "issue": "Pruning (city tree)",
        "category": "Tree Issues",
        "request_volume": int(pruning_stats["request_volume"])
    },
    metadata={
        "group_topic": "Pittsburgh 311 Municipal Service Requests",
        "subtopic": "Parks, Trees, Animals, and Public Facilities",
        "request_type_id": "520",
        "issue": "Pruning (city tree)",
        "category": "Tree Issues",
        "department": "DPW - Forestry Division",
        "record_type": "operational_volume",
        "synthetic": False
    }
)

print("\nPruning volume record:")
print(pruning_volume_record)

print("\nVolume record validation:")
print(validate_record(pruning_volume_record))

# Create a table record for resolution statistics.
pruning_resolution_record = make_record(
    doc_id="parks_pruning_008",
    source_url="https://data.wprdc.org/dataset/311-data/resource/29462525-62a6-45bf-9b5e-ad2e1c06348d",
    license_note="Creative Commons Attribution; subject to WPRDC terms of use.",
    table_json={
        "request_type_id": "520",
        "issue": "Pruning (city tree)",
        "category": "Tree Issues",
        "median_days": float(round(pruning_stats["median_days"], 2)),
        "p75_days": float(round(pruning_stats["p75_days"], 2)),
        "p90_days": float(round(pruning_stats["p90_days"], 2))
    },
    metadata={
        "group_topic": "Pittsburgh 311 Municipal Service Requests",
        "subtopic": "Parks, Trees, Animals, and Public Facilities",
        "request_type_id": "520",
        "issue": "Pruning (city tree)",
        "category": "Tree Issues",
        "department": "DPW - Forestry Division",
        "record_type": "resolution_statistics",
        "synthetic": False
    }
)

print("\nPruning resolution record:")
print(pruning_resolution_record)

print("\nResolution record validation:")
print(validate_record(pruning_resolution_record))

# Combine all pruning records.
all_pruning_records = (
    pruning_records
    + [pruning_volume_record, pruning_resolution_record]
)

print("\nTotal pruning records:", len(all_pruning_records))

for record in all_pruning_records:
    print(record["doc_id"], record["modality"])

import json

# Write the 8 pruning records to a test corpus file.
with open("corpus_test.jsonl", "w", encoding="utf-8") as f:
    for record in all_pruning_records:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

print("\nWrote corpus_test.jsonl")

# Complaint examples for each issue.
complaint_examples = {
    "Pruning (city tree)": {
        "clear": "A city tree in front of my property has several large branches that need pruning.",
        "ambiguous": "The tree by the street is getting too big and needs attention.",
        "incomplete": "There is a tree problem and some branches are hanging down.",
        "typo_heavy": "city tree branchs need prunning they are getting too long",
        "informal": "The street tree is getting kinda out of control. Can someone trim it?",
        "short": "A tree near the sidewalk has several branches that need trimming."
    },

    "Tree Removal": {
        "clear": "A city tree appears dead and unstable and may need to be removed.",
        "ambiguous": "There is a tree near the street that looks unsafe and needs attention.",
        "incomplete": "There is a problem with a tree and it may not be safe.",
        "typo_heavy": "city tree looks ded and probly needs to be removd soon",
        "informal": "This tree looks pretty bad and I think it needs to come down.",
        "short": "A city tree by the street may need to be removed soon."
    },

    "Dead tree (Public property)": {
    "clear": "A dead city tree on public property looks unstable and may be dangerous.",
    "ambiguous": "There is a tree near the street that looks unhealthy and unsafe.",
    "incomplete": "There is a dead-looking tree nearby that may be a problem.",
    "typo_heavy": "dead city tree looks unstabl and might fall down soon",
    "informal": "This tree looks totally dead and kind of dangerous.",
    "short": "A dead city tree near the street may be unsafe soon."
    },

    "Tree Fallen Across Road": {
    "clear": "A city tree has fallen across the road and is blocking traffic.",
    "ambiguous": "A large tree is down near the street and causing a problem.",
    "incomplete": "There is a fallen tree near the road and it needs attention.",
    "typo_heavy": "tree fell accross the road and is blockng cars and traffic",
    "informal": "A tree is down in the street and cars can barely get through.",
    "short": "A fallen tree is blocking part of the road right now."
    },

    "Tree Fallen Across Sidewalk": {
    "clear": "A city tree has fallen across the sidewalk and is blocking pedestrians.",
    "ambiguous": "A tree is down near the sidewalk and people cannot get around it easily.",
    "incomplete": "There is a fallen tree by the sidewalk that needs attention.",
    "typo_heavy": "tree fell accross the sidewak and is blockng people walking",
    "informal": "A tree is down across the sidewalk and people have to walk around it.",
    "short": "A fallen tree is blocking the sidewalk and pedestrian access."
    },

    "Stump Grind/Removal": {
    "clear": "A tree stump on public property needs to be ground down or removed.",
    "ambiguous": "There is an old stump near the street that is causing a problem.",
    "incomplete": "There is a stump left behind and it needs some attention.",
    "typo_heavy": "old tree stump needs grindng or removel from the area",
    "informal": "There is still a big stump here after the tree was taken down.",
    "short": "A tree stump near the street still needs to be removed."
    },

"Planting": {
    "clear": "I would like to request a new city tree to be planted along this street.",
    "ambiguous": "This area could really use another tree near the street.",
    "incomplete": "There are not many trees here and I think another one is needed.",
    "typo_heavy": "can the city plant a new tree here theres not many trees around",
    "informal": "It would be great if the city could put a new tree here.",
    "short": "A new city tree should be planted near this street."
    },

"Parks Trails": {
    "clear": "A park trail is damaged and difficult to use safely.",
    "ambiguous": "There is a problem on one of the trails in the park.",
    "incomplete": "The park trail needs attention because something is wrong.",
    "typo_heavy": "park trail is damagd and hard to walk on safly right now",
    "informal": "The trail in the park is pretty messed up and needs fixing.",
    "short": "A park trail is damaged and needs maintenance soon."
},

"Overgrowth": {
    "clear": "Vegetation in the park is overgrown and blocking part of the walkway.",
    "ambiguous": "The plants in this area are getting out of control.",
    "incomplete": "There is too much overgrowth in this part of the park.",
    "typo_heavy": "weeds and plants are overgrown and blockng the walkway",
    "informal": "The weeds here are getting crazy and covering everything.",
    "short": "Overgrown vegetation is blocking part of the park area."
},

"Other (please describe)": {
    "clear": "There is an animal-related issue that does not fit the listed service types.",
    "ambiguous": "There is some kind of animal problem that needs attention.",
    "incomplete": "There is an animal issue here but I am not sure what category it is.",
    "typo_heavy": "theres some animal problm here not sure what type it is",
    "informal": "There is something going on with an animal and I am not sure what to call it.",
    "short": "There is an animal issue that does not fit the other categories."
},

"Lights": {
    "clear": "A light in the park is not working and the area is very dark at night.",
    "ambiguous": "The park is too dark in this area and needs attention.",
    "incomplete": "There is a lighting problem somewhere in the park.",
    "typo_heavy": "park lite is not workng and its really dark at nite",
    "informal": "One of the park lights is out and this spot is super dark.",
    "short": "A park light is not working and needs to be repaired."
},

"Litter": {
    "clear": "There is a large amount of litter scattered around the park.",
    "ambiguous": "This part of the park is very dirty and needs attention.",
    "incomplete": "There is trash around the park that should be cleaned up.",
    "typo_heavy": "lots of litr and trash all over the park area right now",
    "informal": "There is trash everywhere in this part of the park.",
    "short": "There is litter throughout the park that needs cleanup."
},

"Loose Dog(s)": {
    "clear": "A loose dog is running around the neighborhood without an owner nearby.",
    "ambiguous": "There is a dog wandering around outside by itself.",
    "incomplete": "A dog is loose somewhere nearby and may need help.",
    "typo_heavy": "loose dog runing around with no owner arond anywhere nearby",
    "informal": "There is a dog just roaming around by itself out here.",
    "short": "A loose dog is roaming around without an owner nearby."
},

"Off Leash Exercise Area": {
    "clear": "There is a problem with the designated off-leash dog exercise area.",
    "ambiguous": "The dog area in the park has a problem and needs some attention.",
    "incomplete": "There is an issue at the area for dogs needs attention.",
    "typo_heavy": "off leash dog area has a problm and needs attntion",
    "informal": "Something is wrong with the dog run area at the park.",
    "short": "The off-leash dog exercise area needs maintenance."
},

"Playground": {
    "clear": "Playground equipment in the park is damaged and may be unsafe for children.",
    "ambiguous": "There is a problem with the playground equipment in the park.",
    "incomplete": "Something is wrong with the playground and it needs attention.",
    "typo_heavy": "playground equpment is brokn and might not be safe",
    "informal": "Some of the playground stuff is broken and looks unsafe.",
    "short": "The playground equipment is damaged and needs repair."
},

"Shelters": {
    "clear": "A park shelter is damaged and needs maintenance before it can be used safely.",
    "ambiguous": "There is a problem with one of the shelters in the park.",
    "incomplete": "The park shelter needs some kind of repair or maintenance.",
    "typo_heavy": "park sheltr is damagd and needs to be fixd as soon as posible",
    "informal": "One of the park shelters is in pretty rough shape.",
    "short": "A park shelter is damaged and needs maintenance soon."
},

"Program": {
    "clear": "I have a question or concern about a city parks and recreation program.",
    "ambiguous": "There is an issue with one of the recreation programs.",
    "incomplete": "I need help with a parks and recreation program in the city.",
    "typo_heavy": "i have a problm with a parks rec program and need help",
    "informal": "I have a question about one of the city recreation programs.",
    "short": "I need help with a city parks and recreation program."
},

"Court, Basketball or Tennis": {
    "clear": "A basketball court in the park is damaged and needs to be repaired.",
    "ambiguous": "There is a problem with one of the courts in the park.",
    "incomplete": "The sports court in the park needs some attention.",
    "typo_heavy": "basketball cort is damagd and needs to be repared soon",
    "informal": "The court at the park is in bad shape and needs fixing.",
    "short": "A park sports court is damaged and needs maintenance."
},

"Rodent control": {
    "clear": "There are many rats around this area and rodent control is needed.",
    "ambiguous": "There seem to be a lot of rodents around this location.",
    "incomplete": "There is a rodent problem here that needs attention.",
    "typo_heavy": "lots of rats arond here need rodent contrl help right now",
    "informal": "There are rats all over this area and it is getting bad.",
    "short": "There is a rodent problem in this area that needs control."
},

"Check Conditions": {
    "clear": "Please check the conditions involving an animal at this location.",
    "ambiguous": "There may be an animal problem here that should be checked.",
    "incomplete": "Something involving an animal does not look right here.",
    "typo_heavy": "somthing with an animal looks wrong can somone chek it",
    "informal": "Can someone come check what is going on with this animal?",
    "short": "An animal situation at this location needs to be checked."
},

"City Facility": {
    "clear": "A city facility has a maintenance issue that needs to be inspected and repaired.",
    "ambiguous": "There is a problem with a city building that needs attention.",
    "incomplete": "Something is wrong at a city facility and it needs to be checked.",
    "typo_heavy": "city facilty has a maintnance problm that needs fixng",
    "informal": "Something at this city building is broken and needs to be looked at.",
    "short": "A city facility has a maintenance issue that needs repair."
},

"City Steps, Need Cleared": {
    "clear": "City-owned steps are blocked by debris and need to be cleared.",
    "ambiguous": "The public steps are difficult to use and need attention.",
    "incomplete": "There is something blocking the city steps right now.",
    "typo_heavy": "city steps are blockd and need to be cleard off right now",
    "informal": "The city steps are covered with stuff and hard to walk on.",
    "short": "City-owned steps are blocked and need to be cleared."
},

"Water/Drinking Fountains": {
    "clear": "A public drinking fountain is not working and needs to be repaired.",
    "ambiguous": "There is a problem with the water fountain in this area.",
    "incomplete": "The drinking fountain here is not working correctly.",
    "typo_heavy": "drinking fountin is brokn and no water comes out of it",
    "informal": "The water fountain here is busted and does not work.",
    "short": "A public drinking fountain is broken and needs repair."
},

"Fence": {
    "clear": "A city-owned fence is damaged and needs to be repaired.",
    "ambiguous": "There is a problem with a fence somewhere on city property.",
    "incomplete": "The fence here is damaged and needs some attention.",
    "typo_heavy": "city fence is damagd and needs to be fixd as soon as posible",
    "informal": "This city fence is pretty beat up and needs fixing.",
    "short": "A city-owned fence is damaged and needs repair soon."
},

"Dead Animal": {
    "clear": "There is a dead animal on public property that needs to be removed.",
    "ambiguous": "There is an animal lying outside that does not appear to be alive.",
    "incomplete": "There is a dead animal nearby that needs attention.",
    "typo_heavy": "dead animal on public proerty needs to be removd right now",
    "informal": "There is a dead animal out here that someone needs to pick up.",
    "short": "A dead animal on public property needs to be removed."
},

"Animal Waste": {
    "clear": "There is a large amount of animal waste on public property that needs cleanup.",
    "ambiguous": "There is animal-related waste in this area that needs attention.",
    "incomplete": "There is animal waste here that should be cleaned up.",
    "typo_heavy": "animal waste all over public area needs cleanng right now",
    "informal": "There is animal poop all over this area and it needs cleaning.",
    "short": "Animal waste on public property needs to be cleaned up."
},

"Barking Dog": {
    "clear": "A dog has been barking continuously and causing a noise problem.",
    "ambiguous": "There is a very noisy dog nearby that keeps making noise.",
    "incomplete": "A dog nearby has been making a lot of noise for a while.",
    "typo_heavy": "dog keeps barkng nonstop and is really loud all day",
    "informal": "This dog has been barking forever and it is super loud.",
    "short": "A dog has been barking continuously and causing noise."
},

"Dog License": {
    "clear": "I need information about getting or renewing a city dog license.",
    "ambiguous": "I have a question about paperwork required for my dog.",
    "incomplete": "I need help with something related to my dog's license.",
    "typo_heavy": "need help geting or renewng a dog licens in the city",
    "informal": "I need to figure out how to get a license for my dog.",
    "short": "I need help getting or renewing a city dog license."
    }
}

# Create text records for one issue.
def build_text_records(issue_name, examples):
    issue_info = my_codebook[
        my_codebook["issue"] == issue_name
    ].iloc[0]

    records = []

    for i, (style, text) in enumerate(examples.items(), start=1):
        record = make_record(
            doc_id=f"parks_{issue_info['request_type_id']}_{i:03d}",
            source_url="https://data.wprdc.org/dataset/311-data/resource/6a2c9de6-9cb8-4da6-bd78-0a913cc5790c",
            license_note="Creative Commons Attribution; subject to WPRDC terms of use.",
            raw_text=text,
            metadata={
                "group_topic": "Pittsburgh 311 Municipal Service Requests",
                "subtopic": "Parks, Trees, Animals, and Public Facilities",
                "request_type_id": str(issue_info["request_type_id"]),
                "issue": issue_info["issue"],
                "category": issue_info["category"],
                "department": issue_info["department"],
                "example_style": style,
                "synthetic": True
            }
        )

        records.append(record)

    return records

# Create table records for one issue.
def build_table_records(issue_name):
    stats = issue_stats[
        issue_stats["issue"] == issue_name
    ].iloc[0]

    issue_info = my_codebook[
        my_codebook["issue"] == issue_name
    ].iloc[0]

    volume_record = make_record(
        doc_id=f"parks_{issue_info['request_type_id']}_007",
        source_url="https://data.wprdc.org/dataset/311-data/resource/29462525-62a6-45bf-9b5e-ad2e1c06348d",
        license_note="Creative Commons Attribution; subject to WPRDC terms of use.",
        table_json={
            "request_type_id": str(issue_info["request_type_id"]),
            "issue": issue_info["issue"],
            "category": issue_info["category"],
            "request_volume": int(stats["request_volume"])
        },
        metadata={
            "group_topic": "Pittsburgh 311 Municipal Service Requests",
            "subtopic": "Parks, Trees, Animals, and Public Facilities",
            "request_type_id": str(issue_info["request_type_id"]),
            "issue": issue_info["issue"],
            "category": issue_info["category"],
            "department": issue_info["department"],
            "record_type": "operational_volume",
            "synthetic": False
        }
    )

    resolution_record = make_record(
        doc_id=f"parks_{issue_info['request_type_id']}_008",
        source_url="https://data.wprdc.org/dataset/311-data/resource/29462525-62a6-45bf-9b5e-ad2e1c06348d",
        license_note="Creative Commons Attribution; subject to WPRDC terms of use.",
        table_json={
            "request_type_id": str(issue_info["request_type_id"]),
            "issue": issue_info["issue"],
            "category": issue_info["category"],
            "median_days": float(round(stats["median_days"], 2)),
            "p75_days": float(round(stats["p75_days"], 2)),
            "p90_days": float(round(stats["p90_days"], 2))
        },
        metadata={
            "group_topic": "Pittsburgh 311 Municipal Service Requests",
            "subtopic": "Parks, Trees, Animals, and Public Facilities",
            "request_type_id": str(issue_info["request_type_id"]),
            "issue": issue_info["issue"],
            "category": issue_info["category"],
            "department": issue_info["department"],
            "record_type": "resolution_statistics",
            "synthetic": False
        }
    )

    return [volume_record, resolution_record]

# Validate Tree Removal examples.
print("\nTree Removal validation:")

for i, (style, text) in enumerate(
    complaint_examples["Tree Removal"].items(),
    start=1
):
    record = make_record(
        doc_id=f"parks_tree_removal_{i:03d}",
        source_url="https://data.wprdc.org/dataset/311-data/resource/6a2c9de6-9cb8-4da6-bd78-0a913cc5790c",
        license_note="Creative Commons Attribution; subject to WPRDC terms of use.",
        raw_text=text,
        metadata={
            "group_topic": "Pittsburgh 311 Municipal Service Requests",
            "subtopic": "Parks, Trees, Animals, and Public Facilities",
            "request_type_id": "345",
            "issue": "Tree Removal",
            "category": "Tree Issues",
            "department": "DPW - Forestry Division",
            "example_style": style,
            "synthetic": True
        }
    )

    print(style, validate_record(record))

# Validate Dead tree examples.
print("\nDead tree validation:")

for i, (style, text) in enumerate(
    complaint_examples["Dead tree (Public property)"].items(),
    start=1
):
    record = make_record(
        doc_id=f"parks_dead_tree_{i:03d}",
        source_url="https://data.wprdc.org/dataset/311-data/resource/6a2c9de6-9cb8-4da6-bd78-0a913cc5790c",
        license_note="Creative Commons Attribution; subject to WPRDC terms of use.",
        raw_text=text,
        metadata={
            "group_topic": "Pittsburgh 311 Municipal Service Requests",
            "subtopic": "Parks, Trees, Animals, and Public Facilities",
            "request_type_id": "518",
            "issue": "Dead tree (Public property)",
            "category": "Tree Issues",
            "department": "DPW - Forestry Division",
            "example_style": style,
            "synthetic": True
        }
    )

    print(style, validate_record(record))

# Validate Tree Fallen Across Road examples.
print("\nTree Fallen Across Road validation:")

for i, (style, text) in enumerate(
    complaint_examples["Tree Fallen Across Road"].items(),
    start=1
):
    record = make_record(
        doc_id=f"parks_tree_road_{i:03d}",
        source_url="https://data.wprdc.org/dataset/311-data/resource/6a2c9de6-9cb8-4da6-bd78-0a913cc5790c",
        license_note="Creative Commons Attribution; subject to WPRDC terms of use.",
        raw_text=text,
        metadata={
            "group_topic": "Pittsburgh 311 Municipal Service Requests",
            "subtopic": "Parks, Trees, Animals, and Public Facilities",
            "request_type_id": "341",
            "issue": "Tree Fallen Across Road",
            "category": "Tree Issues",
            "department": "DPW - Street Maintenance",
            "example_style": style,
            "synthetic": True
        }
    )

    print(style, validate_record(record))

# Validate Tree Fallen Across Sidewalk examples.
print("\nTree Fallen Across Sidewalk validation:")

for i, (style, text) in enumerate(
    complaint_examples["Tree Fallen Across Sidewalk"].items(),
    start=1
):
    record = make_record(
        doc_id=f"parks_tree_sidewalk_{i:03d}",
        source_url="https://data.wprdc.org/dataset/311-data/resource/6a2c9de6-9cb8-4da6-bd78-0a913cc5790c",
        license_note="Creative Commons Attribution; subject to WPRDC terms of use.",
        raw_text=text,
        metadata={
            "group_topic": "Pittsburgh 311 Municipal Service Requests",
            "subtopic": "Parks, Trees, Animals, and Public Facilities",
            "request_type_id": "342",
            "issue": "Tree Fallen Across Sidewalk",
            "category": "Tree Issues",
            "department": "DPW - Street Maintenance",
            "example_style": style,
            "synthetic": True
        }
    )

    print(style, validate_record(record))

# Validate Stump Grind/Removal examples.
print("\nStump Grind/Removal validation:")

for i, (style, text) in enumerate(
    complaint_examples["Stump Grind/Removal"].items(),
    start=1
):
    record = make_record(
        doc_id=f"parks_stump_{i:03d}",
        source_url="https://data.wprdc.org/dataset/311-data/resource/6a2c9de6-9cb8-4da6-bd78-0a913cc5790c",
        license_note="Creative Commons Attribution; subject to WPRDC terms of use.",
        raw_text=text,
        metadata={
            "group_topic": "Pittsburgh 311 Municipal Service Requests",
            "subtopic": "Parks, Trees, Animals, and Public Facilities",
            "request_type_id": "35243",
            "issue": "Stump Grind/Removal",
            "category": "Tree Issues",
            "department": "DPW - Forestry Division",
            "example_style": style,
            "synthetic": True
        }
    )

    print(style, validate_record(record))

# Validate all complaint examples.
print("\nAll complaint examples validation:")

for issue_name, examples in complaint_examples.items():
    print(f"\n{issue_name}")

    records = build_text_records(
        issue_name,
        examples
    )

    for record in records:
        print(
            record["metadata"]["example_style"],
            validate_record(record)
        )

# Test the reusable text-record function.
test_pruning_records = build_text_records(
    "Pruning (city tree)",
    complaint_examples["Pruning (city tree)"]
)

test_tree_removal_records = build_text_records(
    "Tree Removal",
    complaint_examples["Tree Removal"]
)

print("\nFunction test:")
print("Pruning records:", len(test_pruning_records))
print("Tree Removal records:", len(test_tree_removal_records))

print("\nValidation:")
for record in test_pruning_records + test_tree_removal_records:
    print(record["doc_id"], validate_record(record))

# Test the reusable table-record function.
test_pruning_table_records = build_table_records(
    "Pruning (city tree)"
)

print("\nTable function test:")
print("Pruning table records:", len(test_pruning_table_records))

for record in test_pruning_table_records:
    print(record["doc_id"], record["modality"], validate_record(record))

# Create all records for one issue.
# build_issue_records("Pruning (city tree)")
def build_issue_records(issue_name):
    text_records = build_text_records(
        issue_name,
        complaint_examples[issue_name]
    )

    table_records = build_table_records(issue_name)

    return text_records + table_records

# Test the full issue-record function.
test_full_pruning = build_issue_records(
    "Pruning (city tree)"
)

print("\nFull issue function test:")
print("Total records:", len(test_full_pruning))

for record in test_full_pruning:
    print(
        record["doc_id"],
        record["modality"],
        validate_record(record)
    )

# Build the full corpus.
all_records = []

for issue_name in complaint_examples:
    issue_records = build_issue_records(issue_name)
    all_records.extend(issue_records)

print("\nFull corpus size:", len(all_records))

text_count = sum(
    1 for record in all_records
    if record["modality"] == "text"
)

table_count = sum(
    1 for record in all_records
    if record["modality"] == "table"
)

print("Text records:", text_count)
print("Table records:", table_count)

# Write the final corpus.
with open("corpus.jsonl", "w", encoding="utf-8") as f:
    for record in all_records:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

print("\nWrote corpus.jsonl")