from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

ROOT = Path(__file__).resolve().parents[1]
df = pd.read_csv(ROOT / "data" / "student_performance_500.csv")
OUT = ROOT / "app" / "static" / "charts"
OUT.mkdir(parents=True, exist_ok=True)

# Dark, colorful chart theme for the web dashboard.
BG = "#111827"
PANEL = "#182235"
TEXT = "#E5E7EB"
GRID = "#334155"
BLUE = "#38BDF8"
PURPLE = "#A78BFA"
PINK = "#F472B6"
GREEN = "#34D399"
YELLOW = "#FBBF24"
ORANGE = "#FB923C"
RED = "#FB7185"
TEAL = "#2DD4BF"
PALETTE = [BLUE, PURPLE, PINK, GREEN, YELLOW, ORANGE, RED, TEAL]

sns.set_theme(style="darkgrid", rc={
    "figure.facecolor": BG,
    "axes.facecolor": PANEL,
    "axes.edgecolor": GRID,
    "axes.labelcolor": TEXT,
    "axes.titlecolor": TEXT,
    "xtick.color": TEXT,
    "ytick.color": TEXT,
    "grid.color": GRID,
    "text.color": TEXT,
    "font.size": 10,
})

def finish(ax, title):
    ax.set_title(title, fontsize=13, fontweight="bold", pad=12)
    ax.figure.patch.set_facecolor(BG)
    ax.set_facecolor(PANEL)
    for spine in ax.spines.values():
        spine.set_color(GRID)
    ax.grid(True, color=GRID, alpha=.55, linewidth=.8)
    ax.tick_params(colors=TEXT)
    ax.xaxis.label.set_color(TEXT)
    ax.yaxis.label.set_color(TEXT)
    ax.figure.tight_layout(pad=1.2)

# 1. Department-wise average score - colorful bar chart
dept = df.groupby("department")["final_score"].mean().sort_values(ascending=False)
fig, ax = plt.subplots(figsize=(7, 4), facecolor=BG)
ax.bar(dept.index, dept.values, color=PALETTE[:len(dept)], edgecolor="#0F172A", linewidth=1.2)
for i, v in enumerate(dept.values):
    ax.text(i, v + 1, f"{v:.1f}", ha="center", color=TEXT, fontweight="bold")
ax.set_xlabel("Department")
ax.set_ylabel("Average Final Score")
finish(ax, "Department-wise Average Final Score")
fig.savefig(OUT / "department_average.png", dpi=160, facecolor=BG, bbox_inches="tight")
plt.close(fig)

# 2. Study hours vs score - cyan scatter + pink regression line
fig, ax = plt.subplots(figsize=(7, 4), facecolor=BG)
sns.scatterplot(data=df, x="study_hours_per_day", y="final_score", color=BLUE, alpha=.72, s=42, edgecolor="#E0F2FE", linewidth=.25, ax=ax)
sns.regplot(data=df, x="study_hours_per_day", y="final_score", scatter=False, ci=None, color=PINK, line_kws={"linewidth": 2.5}, ax=ax)
ax.set_xlabel("Study Hours per Day")
ax.set_ylabel("Final Score")
finish(ax, "Study Hours vs Final Score")
fig.savefig(OUT / "study_vs_score.png", dpi=160, facecolor=BG, bbox_inches="tight")
plt.close(fig)

# 3. Attendance vs score - green scatter + yellow regression line
fig, ax = plt.subplots(figsize=(7, 4), facecolor=BG)
sns.scatterplot(data=df, x="attendance_percentage", y="final_score", color=GREEN, alpha=.72, s=42, edgecolor="#DCFCE7", linewidth=.25, ax=ax)
sns.regplot(data=df, x="attendance_percentage", y="final_score", scatter=False, ci=None, color=YELLOW, line_kws={"linewidth": 2.5}, ax=ax)
ax.set_xlabel("Attendance (%)")
ax.set_ylabel("Final Score")
finish(ax, "Attendance vs Final Score")
fig.savefig(OUT / "attendance_vs_score.png", dpi=160, facecolor=BG, bbox_inches="tight")
plt.close(fig)

# 4. Semester trend - purple line with colored markers
sem = df.groupby("semester")["final_score"].mean()
fig, ax = plt.subplots(figsize=(7, 4), facecolor=BG)
ax.plot(sem.index, sem.values, color=PURPLE, linewidth=3, marker="o", markersize=8, markerfacecolor=TEAL, markeredgecolor="#ECFEFF", markeredgewidth=1.2)
for x, y in zip(sem.index, sem.values):
    ax.annotate(f"{y:.1f}", (x, y), textcoords="offset points", xytext=(0, 9), ha="center", color=TEXT, fontweight="bold")
ax.set_xlabel("Semester")
ax.set_ylabel("Average Final Score")
finish(ax, "Semester-wise Average Final Score")
fig.savefig(OUT / "semester_trend.png", dpi=160, facecolor=BG, bbox_inches="tight")
plt.close(fig)

# 5. Score distribution - colorful histogram
fig, ax = plt.subplots(figsize=(7, 4), facecolor=BG)
ax.hist(df["final_score"], bins=10, color=BLUE, edgecolor="#E0F2FE", linewidth=1.0, alpha=.9)
ax.axvline(df["final_score"].mean(), color=PINK, linestyle="--", linewidth=2.5, label=f"Mean: {df['final_score'].mean():.1f}")
ax.set_xlabel("Final Score")
ax.set_ylabel("Number of Students")
ax.legend(facecolor=PANEL, edgecolor=GRID, labelcolor=TEXT)
finish(ax, "Final Score Distribution")
fig.savefig(OUT / "score_distribution.png", dpi=160, facecolor=BG, bbox_inches="tight")
plt.close(fig)

# 6. Performance category - colorful donut chart
cats = pd.cut(df["final_score"], bins=[0, 50, 60, 75, 100], labels=["At Risk", "Average", "Good", "Excellent"], include_lowest=True)
counts = cats.value_counts().reindex(["Excellent", "Good", "Average", "At Risk"]).fillna(0)
fig, ax = plt.subplots(figsize=(6, 6), facecolor=BG)
wedges, texts, autotexts = ax.pie(counts, labels=counts.index, autopct="%1.0f%%", startangle=90, colors=[GREEN, BLUE, YELLOW, RED], pctdistance=.76, wedgeprops={"width": .42, "edgecolor": BG, "linewidth": 2})
for t in texts:
    t.set_color(TEXT)
    t.set_fontweight("bold")
for t in autotexts:
    t.set_color("#FFFFFF")
    t.set_fontweight("bold")
ax.set_title("Performance Category", color=TEXT, fontsize=13, fontweight="bold", pad=12)
fig.tight_layout()
fig.savefig(OUT / "performance_category.png", dpi=160, facecolor=BG, bbox_inches="tight")
plt.close(fig)

# 7. Correlation heatmap - vivid diverging palette
corr_cols = ["attendance_percentage", "study_hours_per_day", "previous_exam_score", "midterm_score", "assignment_percentage", "practical_lab_score", "sleep_hours_per_day", "online_learning_hours", "previous_backlogs", "quiz_score", "class_participation_score", "final_score"]
fig, ax = plt.subplots(figsize=(10, 8), facecolor=BG)
sns.heatmap(df[corr_cols].corr(), annot=True, fmt=".2f", cmap="magma", center=0, linewidths=.6, linecolor=BG, cbar_kws={"shrink": .8}, annot_kws={"fontsize": 8}, ax=ax)
ax.set_title("Academic Feature Correlation Heatmap", color=TEXT, fontsize=13, fontweight="bold", pad=12)
ax.tick_params(axis="x", rotation=45, colors=TEXT)
ax.tick_params(axis="y", rotation=0, colors=TEXT)
fig.tight_layout()
fig.savefig(OUT / "correlation_heatmap.png", dpi=160, facecolor=BG, bbox_inches="tight")
plt.close(fig)

print("Dark-mode colorful charts generated successfully.")
