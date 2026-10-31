"""Figure helpers for the Instagram engagement analysis. Run directly to export all PNGs."""
from pathlib import Path
import matplotlib
matplotlib.use("Agg") if __name__ == "__main__" else None
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import pandas as pd
import seaborn as sns

ROOT = Path(__file__).resolve().parents[1]
IMG = ROOT / "images"

NAVY, TEAL, ORANGE, GREY, RED = "#1F3A5F", "#2A9D8F", "#E76F51", "#9AA5B1", "#C0392B"
SOURCE_COLORS = {"from_home_pct": NAVY, "from_hashtags_pct": TEAL,
                 "from_explore_pct": ORANGE, "from_other_pct": GREY}
SOURCE_LABELS = {"from_home_pct": "Home feed", "from_hashtags_pct": "Hashtags",
                 "from_explore_pct": "Explore", "from_other_pct": "Other"}

plt.rcParams.update({
    "figure.dpi": 110, "savefig.dpi": 160, "savefig.bbox": "tight",
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.titleweight": "bold", "axes.titlesize": 12, "axes.labelsize": 10,
    "font.size": 10, "axes.grid": True, "grid.alpha": 0.25, "axes.axisbelow": True,
})


def _save(fig, name):
    IMG.mkdir(exist_ok=True)
    fig.savefig(IMG / name)
    return fig


def impressions_distribution(df):
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].hist(df["impressions"], bins=25, color=NAVY, edgecolor="white")
    med = df["impressions"].median()
    ax[0].axvline(med, color=ORANGE, lw=2, ls="--", label=f"Median {med:,.0f}")
    ax[0].set(title="Impressions per post", xlabel="Impressions", ylabel="Posts")
    ax[0].legend()
    s = df["impressions"].sort_values(ascending=False).reset_index(drop=True)
    cum = s.cumsum() / s.sum() * 100
    ax[1].plot((np.arange(len(s)) + 1) / len(s) * 100, cum, color=TEAL, lw=2.5)
    ax[1].plot([0, 100], [0, 100], color=GREY, ls=":")
    ax[1].set(title="Reach is concentrated in a few posts",
              xlabel="% of posts (best first)", ylabel="Cumulative % of impressions")
    ax[1].xaxis.set_major_formatter(mtick.PercentFormatter())
    ax[1].yaxis.set_major_formatter(mtick.PercentFormatter())
    top10 = cum.iloc[int(round(len(s) * 0.10)) - 1]
    ax[1].annotate(f"Top 10% of posts\n= {top10:.0f}% of impressions", xy=(10, top10),
                   xytext=(30, top10 - 22), arrowprops=dict(arrowstyle="->", color=GREY))
    fig.tight_layout()
    return _save(fig, "01_impressions_distribution.png")


def traffic_sources(df):
    cols = list(SOURCE_COLORS)
    order = ["Low", "Medium", "High"]
    mix = df.groupby("impression_tier", observed=True)[cols].mean().loc[order]
    overall = df[cols].mean().to_frame("All posts").T
    mix = pd.concat([overall, mix])
    fig, ax = plt.subplots(figsize=(9, 4))
    left = np.zeros(len(mix))
    for c in cols:
        ax.barh(mix.index, mix[c], left=left, color=SOURCE_COLORS[c], label=SOURCE_LABELS[c])
        for y, (l, v) in enumerate(zip(left, mix[c])):
            if v > 6:
                ax.text(l + v / 2, y, f"{v:.0f}%", ha="center", va="center", color="white", fontsize=9)
        left += mix[c].values
    ax.set(title="Where impressions come from, by post performance tier", xlabel="Share of impressions (%)")
    ax.invert_yaxis(); ax.grid(False)
    ax.legend(ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.18), frameon=False)
    fig.tight_layout()
    return _save(fig, "02_traffic_sources.png")


def engagement_overview(df):
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    mix = df[["likes", "saves", "shares", "comments"]].sum()
    mix = (mix / mix.sum() * 100).sort_values()
    ax[0].barh(mix.index.str.title(), mix.values, color=[GREY, GREY, TEAL, NAVY])
    for i, v in enumerate(mix.values):
        ax[0].text(v + 0.8, i, f"{v:.1f}%", va="center")
    ax[0].set(title="What the interactions are made of", xlabel="Share of all interactions (%)", xlim=(0, 60))
    ax[0].grid(False)
    ax[1].hist(df["engagement_rate"], bins=20, color=TEAL, edgecolor="white")
    med = df["engagement_rate"].median()
    ax[1].axvline(med, color=ORANGE, lw=2, ls="--", label=f"Median {med:.1f}%")
    ax[1].set(title="Engagement rate per post", xlabel="(likes+comments+shares+saves) / impressions (%)", ylabel="Posts")
    ax[1].legend()
    fig.tight_layout()
    return _save(fig, "03_engagement_overview.png")


def topic_performance(df):
    g = (df.groupby("topic").agg(posts=("post_id", "count"),
                                 impressions=("impressions", "median"),
                                 er=("engagement_rate", "median"),
                                 pv=("profile_visit_rate", "median"))
         .sort_values("impressions"))
    fig, ax = plt.subplots(1, 3, figsize=(14, 4.4), sharey=True)
    for a, col, title, color, fmt in [
        (ax[0], "impressions", "Median impressions", NAVY, "{:,.0f}"),
        (ax[1], "er", "Median engagement rate (%)", TEAL, "{:.1f}%"),
        (ax[2], "pv", "Median profile-visit rate (%)", ORANGE, "{:.2f}%")]:
        a.barh(g.index, g[col], color=color)
        for i, v in enumerate(g[col]):
            a.text(v, i, " " + fmt.format(v), va="center", fontsize=9)
        a.set_title(title); a.grid(False)
        a.set_xlim(0, g[col].max() * 1.25)
    ax[0].set_yticks(range(len(g)))
    ax[0].set_yticklabels([f"{t} (n={n})" for t, n in zip(g.index, g["posts"])])
    fig.suptitle("Content topics: reach, engagement and profile-visit rate", fontweight="bold", y=1.02)
    fig.tight_layout()
    return _save(fig, "04_topic_performance.png")


def explore_vs_impressions(df):
    fig, ax = plt.subplots(figsize=(8, 5))
    top = df["viral_flag"]
    ax.scatter(df.loc[~top, "from_explore_pct"], df.loc[~top, "impressions"], s=45, color=GREY, alpha=.8, label="Other posts")
    ax.scatter(df.loc[top, "from_explore_pct"], df.loc[top, "impressions"], s=70, color=ORANGE, edgecolor="white", label="Top 10% by impressions")
    ax.set_yscale("log")
    ax.set(title="Posts that break out are pushed by Explore",
           xlabel="Share of impressions from Explore (%)", ylabel="Impressions (log scale)")
    ax.legend()
    fig.tight_layout()
    return _save(fig, "05_explore_vs_impressions.png")


def hashtag_and_caption(df):
    d = df.copy()
    d["hashtag_bucket"] = pd.cut(d["hashtag_count"], [0, 15, 20, 40], labels=["10-15", "16-20", "21+"])
    d["caption_bucket"] = pd.qcut(d["caption_length"], 3, labels=["Short", "Medium", "Long"])
    fig, ax = plt.subplots(1, 3, figsize=(14, 4.2))
    sns.boxplot(data=d, x="hashtag_bucket", y="impressions", color=NAVY, ax=ax[0], showfliers=False, boxprops=dict(alpha=.85))
    ax[0].set(title="Impressions by # of hashtags", xlabel="Hashtags per post", ylabel="Impressions")
    sns.boxplot(data=d, x="caption_bucket", y="engagement_rate", color=TEAL, ax=ax[1], showfliers=False, boxprops=dict(alpha=.85))
    ax[1].set(title="Engagement rate by caption length", xlabel="Caption length (terciles)", ylabel="Engagement rate (%)")
    sns.boxplot(data=d, x="caption_bucket", y="saves", color=ORANGE, ax=ax[2], showfliers=False, boxprops=dict(alpha=.85))
    ax[2].set(title="Saves by caption length", xlabel="Caption length (terciles)", ylabel="Saves")
    fig.tight_layout()
    return _save(fig, "06_hashtags_and_captions.png")


def link_in_bio(df):
    metrics = [("impressions", "Median impressions", "{:,.0f}"),
               ("engagement_rate", "Median engagement rate (%)", "{:.1f}"),
               ("profile_visit_rate", "Median profile-visit rate (%)", "{:.2f}"),
               ("follows", "Median follows per post", "{:.0f}")]
    g = df.groupby("has_link_in_bio")[[m for m, _, _ in metrics]].median()
    fig, ax = plt.subplots(1, 4, figsize=(14, 3.6))
    for a, (m, title, fmt) in zip(ax, metrics):
        vals = [g.loc[False, m], g.loc[True, m]]
        a.bar(["No CTA", "'Link in bio'"], vals, color=[GREY, ORANGE])
        for i, v in enumerate(vals):
            a.text(i, v, fmt.format(v), ha="center", va="bottom", fontsize=10)
        a.set_title(title, fontsize=10); a.grid(False)
        a.set_ylim(0, max(vals) * 1.2)
    fig.suptitle(f"Posts with a 'Link in bio' call-to-action (n={int(df['has_link_in_bio'].sum())}) vs the rest (n={int((~df['has_link_in_bio']).sum())})", fontweight="bold", y=1.04)
    fig.tight_layout()
    return _save(fig, "07_link_in_bio.png")


def funnel(df):
    steps = [("Impressions", df["impressions"].sum()), ("Profile visits", df["profile_visits"].sum()), ("New follows", df["follows"].sum())]
    fig, ax = plt.subplots(figsize=(8, 3.6))
    colors = [NAVY, TEAL, ORANGE]
    ax.barh([s for s, _ in steps][::-1], [np.log10(v) for _, v in steps][::-1], color=colors[::-1])
    for i, (s, v) in enumerate(steps[::-1]):
        ax.text(np.log10(v) + .05, i, f"{v:,.0f}", va="center", fontweight="bold")
    pv = steps[1][1] / steps[0][1] * 100
    fl = steps[2][1] / steps[1][1] * 100
    ax.set(title=f"Impression -> profile visit ({pv:.2f}%) -> follow ({fl:.1f}% of visits)", xlabel="log10 scale", xlim=(0, 7.2))
    ax.grid(False); ax.set_xticks([])
    fig.tight_layout()
    return _save(fig, "08_funnel.png")


def correlation_heatmap(df):
    cols = ["impressions", "from_home", "from_hashtags", "from_explore", "likes", "saves", "shares",
            "comments", "profile_visits", "follows", "hashtag_count", "caption_length"]
    corr = df[cols].corr(method="spearman")
    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdBu_r", center=0, vmin=-1, vmax=1,
                linewidths=.5, cbar_kws={"shrink": .8}, ax=ax, annot_kws={"size": 8})
    ax.set_title("Spearman rank correlation between post metrics"); ax.grid(False)
    fig.tight_layout()
    return _save(fig, "09_correlation_heatmap.png")


ALL = [impressions_distribution, traffic_sources, engagement_overview, topic_performance,
       explore_vs_impressions, hashtag_and_caption, link_in_bio, funnel, correlation_heatmap]


def make_all(df=None):
    if df is None:
        df = pd.read_csv(ROOT / "data" / "processed" / "instagram_clean.csv")
    for fn in ALL:
        fn(df); plt.close("all")
    print(f"saved {len(ALL)} figures to {IMG.relative_to(ROOT)}/")


if __name__ == "__main__":
    make_all()
