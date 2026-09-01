from PIL import Image, ImageDraw, ImageFont
import os

out_dir = os.path.join("images")
os.makedirs(out_dir, exist_ok=True)
img = Image.new("RGB", (1400, 900), "#0b1220")
draw = ImageDraw.Draw(img)

# Layout blocks
sidebar = (0, 0, 360, 900)
main = (360, 0, 1400, 900)
draw.rectangle(sidebar, fill="#171d2b")
draw.rectangle(main, fill="#0b1220")

# Fonts
font_bold = ImageFont.truetype("C:/Windows/Fonts/segoeuib.ttf", 30)
font_med = ImageFont.truetype("C:/Windows/Fonts/segoeui.ttf", 22)
font_small = ImageFont.truetype("C:/Windows/Fonts/segoeui.ttf", 18)
header_font = ImageFont.truetype("C:/Windows/Fonts/segoeui.ttf", 58)
sub_font = ImageFont.truetype("C:/Windows/Fonts/segoeui.ttf", 28)
small_font = ImageFont.truetype("C:/Windows/Fonts/segoeui.ttf", 20)
bar_font = ImageFont.truetype("C:/Windows/Fonts/segoeui.ttf", 20)

# Sidebar filters
label_x = 30
label_y = 20
draw.text((label_x, label_y), "Filter Customers", fill="white", font=font_bold)

section_y = 90
for label, options in [
    ("Reward Tier", ["Gold", "Platinum"]),
    ("Region", ["Central", "East", "Metro", "North"]),
    ("Card Type", ["Gold", "Platinum", "Silver", "Titanium"]),
]:
    draw.text((25, section_y), label, fill="#d6e1f2", font=font_med)
    y = section_y + 32
    for opt in options:
        draw.rounded_rectangle((25, y, 250, y + 38), radius=12, fill="#e74c3c")
        draw.text((43, y + 7), opt, fill="white", font=font_small)
        y += 48
    section_y = y + 28

# Main header
# Credit card icon
icon_box = (420, 45, 500, 105)
draw.rounded_rectangle(icon_box, radius=12, fill="#4cc5ff")
draw.rounded_rectangle((450, 56, 490, 88), radius=6, fill="#eaf6ff")

draw.text((520, 35), "Bank Customer", fill="white", font=header_font)
draw.text((520, 95), "Rewards Dashboard", fill="white", font=header_font)
draw.text((420, 180), "Reward and discount segmentation for credit-card customers", fill="#b9c8db", font=sub_font)

draw.text((420, 240), "Overview", fill="white", font=header_font)

# KPI cards
kpis = [
    ("Total Customers", "8,000"),
    ("Platinum Customers", "3156"),
    ("Average Monthly Spend", "$2,025"),
    ("Average Transactions", "32"),
]
start_x = 420
y = 330
card_w = 240
card_h = 100
for i, (label, value) in enumerate(kpis):
    x = start_x + i * 245
    draw.rounded_rectangle((x, y, x + card_w, y + card_h), radius=12, fill="#0f1d2d", outline="#1f3a4d")
    draw.text((x + 16, y + 14), label, fill="#b9c8db", font=small_font)
    draw.text((x + 16, y + 42), value, fill="white", font=ImageFont.truetype("C:/Windows/Fonts/segoeui.ttf", 36))

# Bar chart panel
bar_x0, bar_y0, bar_x1, bar_y1 = (420, 470, 650, 760)
draw.rounded_rectangle((bar_x0, bar_y0, bar_x1, bar_y1), radius=12, fill="#0f1d2d", outline="#1f3a4d")
for idx, val in enumerate([1, 2, 3]):
    x = 455 + idx * 65
    height = 120 + idx * 35
    yb = 640 - height
    draw.rectangle((x, yb, x + 50, 640), fill="#9fd1ea")
    draw.text((x + 2, 648), ["Gold", "Platinum", "Silver"][idx], fill="white", font=bar_font)

# Scatter panel
plot_x0, plot_y0, plot_x1, plot_y1 = (700, 560, 1260, 840)
draw.text((700, 510), "Customer Value vs Transactions", fill="white", font=ImageFont.truetype("C:/Windows/Fonts/segoeui.ttf", 28))
draw.rounded_rectangle((plot_x0, plot_y0, plot_x1, plot_y1), radius=12, fill="#0f1d2d", outline="#1f3a4d")
# Axes
x_axis = (plot_x0 + 25, plot_y1 - 25, plot_x1 - 25, plot_y1 - 25)
y_axis = (plot_x0 + 25, plot_y1 - 25, plot_x0 + 25, plot_y0 + 25)
draw.line(x_axis, fill="#d7e2ee", width=2)
draw.line(y_axis, fill="#d7e2ee", width=2)
# Scatter points
for i in range(80):
    x = plot_x0 + 30 + (i % 20) * 28 + (i // 20) * 6
    y = plot_y1 - 30 - (i % 5) * 18 - (i // 5) * 14
    color = "#d8a7ff" if i % 2 == 0 else "#7ec7ff"
    draw.ellipse((x, y, x + 6, y + 6), fill=color)

# Legend
legend_x = 1130
legend_y = 620
for name, color in [("Platinum", "#d8a7ff"), ("Gold", "#7ec7ff")]:
    draw.ellipse((legend_x, legend_y, legend_x + 18, legend_y + 18), fill=color)
    draw.text((legend_x + 28, legend_y - 2), name, fill="white", font=bar_font)
    legend_y += 34

# Save file
output_path = os.path.join(out_dir, "bank_customer_rewards_dashboard.png")
img.save(output_path)
print(f"Created {output_path}")
