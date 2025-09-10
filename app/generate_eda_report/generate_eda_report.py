import pandas as pd
from pathlib import Path
import base64


# --- Helper function to create a Markdown image tag from base64 data ---
def create_markdown_image(
    encoded_str, alt_text="Image", width=None, height=None
):
    """
    Generates a Markdown image tag with a base64-encoded image.
    If width and height are None, the image will be displayed at its original size.
    """
    if isinstance(encoded_str, str) and encoded_str:
        width_attr = f'width="{width}"' if width is not None else ""
        height_attr = f'height="{height}"' if height is not None else ""
        return f'<img src="data:image/jpeg;base64,{encoded_str}" alt="{alt_text}" {width_attr} {height_attr} style="border-radius: 8px;">'
    return ""


def generate_eda_report(df: pd.DataFrame, output_path: str = "eda_report.md"):
    """
    Performs EDA on the face matching results and generates a Markdown report.

    Args:
        df (pd.DataFrame): The input DataFrame.
        output_path (str): The path to save the output Markdown file.
    """
    if df.empty:
        print("Error: The input DataFrame is empty.")
        return

    # --- 1. How much frames each character/actor appears ---
    actor_counts = df["name"].value_counts()

    actor_counts_markdown = ""
    if "profile_image_base64" in df.columns:
        # Get a unique list of actors to find their profile images
        unique_actors = (
            df[["name", "profile_image_base64"]]
            .drop_duplicates(subset=["name"])
            .set_index("name")
        )

        for actor, count in actor_counts.items():
            base64_img = unique_actors.loc[actor]["profile_image_base64"]
            img_tag = create_markdown_image(
                base64_img, alt_text=actor, width=100
            )
            actor_counts_markdown += (
                f"- {img_tag} **{actor}**: Appeared for **{count}** seconds.\n"
            )
    else:
        for actor, count in actor_counts.items():
            actor_counts_markdown += (
                f"- **{actor}**: Appeared for **{count}** seconds.\n"
            )
        actor_counts_markdown += "\n*Note: Actor profile images could not be included as the required column was missing.*\n"

    # --- 2. Best and worst scores ---
    best_match = df.loc[df["similarity_score"].idxmax()]
    worst_match = df.loc[df["similarity_score"].idxmin()]

    # best_match_img = create_markdown_image(
    #     best_match["scene_image_base64"],
    #     alt_text="Best Match Scene",
    #     width=480,
    #     height=270,
    # )
    # worst_match_img = create_markdown_image(
    #     worst_match["scene_image_base64"],
    #     alt_text="Worst Match Scene",
    #     width=480,
    #     height=270,
    # )

    # --- 3. Some interesting facts ---
    avg_score = df["similarity_score"].mean()

    # Categorize scores into confidence bins
    bins = [0, 0.5, 0.7, 0.9, 1.0]
    labels = [
        "Weak (<0.5)",
        "Moderate (0.5-0.7)",
        "Good (0.7-0.9)",
        "Excellent (0.9-1.0)",
    ]
    df["confidence_bin"] = pd.cut(
        df["similarity_score"], bins=bins, labels=labels, right=False
    )
    confidence_counts = df["confidence_bin"].value_counts().sort_index()
    total_matches = len(df)

    confidence_markdown = ""
    for label, count in confidence_counts.items():
        percentage = (count / total_matches) * 100
        confidence_markdown += (
            f"- **{label}**: {count} matches ({percentage:.2f}%)\n"
        )

    # --- Construct the final Markdown content ---
    markdown_content = f"""
# 🎬 Face Matching Analysis Report

## 📊 Overall Statistics
- **Total Matched Frames**: {len(df)}
- **Average Similarity Score**: {avg_score:.4f}

---

## 🏆 Best and Worst Matches

### Best Match
The model's most confident match was with a score of **{best_match['similarity_score']:.4f}**.
- **Character**: {best_match['character']} (as **{best_match['name']}**)
- **Original Frame**: ##best_match_img

### Worst Match
The least confident match had a score of **{worst_match['similarity_score']:.4f}**.
- **Character**: {worst_match['character']} (as **{worst_match['name']}**)
- **Original Frame**: ##worst_match_img

---

## 👤 Character Screen Time
This section shows how many seconds each actor appeared on screen.

{actor_counts_markdown}

---

## 🧐 Confidence Breakdown
This is how the matches are distributed based on the similarity score.

{confidence_markdown}
"""

    # Save the Markdown file
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(markdown_content)

    print(f"\n✅ Report generated successfully! Find it at '{output_path}'")
