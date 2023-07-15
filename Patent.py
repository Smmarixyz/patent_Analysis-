import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import tkinter as tk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from tkinter import ttk
import numpy as np
import scipy.stats as stats

# Sample patent data
data = [
    {"patent_id": 1, "title": "System and method for image recognition"},
    {"patent_id": 2, "title": "Machine learning algorithms for natural language processing"},
    {"patent_id": 3, "title": "Improved techniques for data compression"},
    {"patent_id": 4, "title": "Methods and systems for autonomous vehicles"},
    {"patent_id": 5, "title": "Image processing using convolutional neural networks"},
    {"patent_id": 6, "title": "Speech recognition system with deep learning"},
    {"patent_id": 7, "title": "Efficient data storage and retrieval methods"},
    {"patent_id": 8, "title": "Systems and methods for personalized recommendation"},
    {"patent_id": 9, "title": "Advancements in renewable energy generation"},
    {"patent_id": 10, "title": "Artificial intelligence for healthcare diagnostics"},
]

# Convert data to a DataFrame
df = pd.DataFrame(data)

# Text preprocessing and feature extraction
vectorizer = TfidfVectorizer(stop_words='english')
X = vectorizer.fit_transform(df['title'])

# Perform Latent Dirichlet Allocation (LDA) for topic modeling
num_topics = 3
lda = LatentDirichletAllocation(n_components=num_topics, random_state=42)
lda.fit(X)

# Get the top N words per topic
num_words = 5
feature_names = vectorizer.get_feature_names_out()
topics = []
for topic_idx, topic in enumerate(lda.components_):
    top_words = [feature_names[i] for i in topic.argsort()[:-num_words - 1:-1]]
    topics.append(top_words)

# Create a Tkinter window
window = tk.Tk()
window.title("Patent Analysis")

# Create labels to display topics and associated top words
topic_labels = []
for topic_idx, top_words in enumerate(topics):
    label = tk.Label(window, text=f"Topic {topic_idx + 1}: {', '.join(top_words)}")
    label.grid(row=0, column=topic_idx, padx=10, pady=10)
    topic_labels.append(label)

# Calculate the topic distribution DataFrame
topic_distribution = lda.transform(X)
topic_names = [f"Topic {i + 1}" for i in range(num_topics)]
df_topic_distribution = pd.DataFrame(topic_distribution, columns=topic_names)

# Create a frame for the plots
plot_frame = tk.Frame(window)
plot_frame.grid(row=1, column=0, columnspan=num_topics, padx=10, pady=10)

# Plotting the topic distribution
fig_topic_dist = plt.Figure(figsize=(4, 2))
ax_topic_dist = fig_topic_dist.add_subplot(111)
sns.barplot(x=df_topic_distribution.columns, y=df_topic_distribution.sum(), color='steelblue', ax=ax_topic_dist)
ax_topic_dist.set_xlabel('Topic')
ax_topic_dist.set_ylabel('Number of Patents')
ax_topic_dist.set_title('Topic Distribution in Sample Patents')
canvas_topic_dist = FigureCanvasTkAgg(fig_topic_dist, master=plot_frame)
canvas_topic_dist.draw()
canvas_topic_dist.get_tk_widget().pack(side=tk.RIGHT, padx=10)

# Plotting the patent count per topic
fig_patent_count = plt.Figure(figsize=(4, 2))
ax_patent_count = fig_patent_count.add_subplot(111)
sns.barplot(x=topic_names, y=df_topic_distribution.sum(), color='steelblue', ax=ax_patent_count)
ax_patent_count.set_xlabel('Topic')
ax_patent_count.set_ylabel('Number of Patents')
ax_patent_count.set_title('Patent Count per Topic')
canvas_patent_count = FigureCanvasTkAgg(fig_patent_count, master=plot_frame)
canvas_patent_count.draw()
canvas_patent_count.get_tk_widget().pack(side=tk.LEFT, padx=10)

# Create a frame for the word clouds
wordcloud_frame = tk.Frame(window)
wordcloud_frame.grid(row=2, column=0, columnspan=num_topics, padx=10, pady=10)

# Create a canvas and scrollbar for the word clouds
canvas_wordcloud = tk.Canvas(wordcloud_frame, width=1000, height=200, scrollregion=(0, 0, 1000, 200))
scrollbar_wordcloud = ttk.Scrollbar(wordcloud_frame, orient=tk.HORIZONTAL, command=canvas_wordcloud.xview)

scrollbar_wordcloud.pack(side=tk.BOTTOM, fill=tk.X)
scrollbar_wordcloud.pack(side=tk.BOTTOM, fill=tk.Y)
canvas_wordcloud.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
canvas_wordcloud.configure(xscrollcommand=scrollbar_wordcloud.set)

# Plotting the word cloud for each topic
x_pos = 0
for topic_idx, top_words in enumerate(topics):
    wordcloud = WordCloud(background_color='white').generate(' '.join(top_words))
    wordcloud_figure = plt.Figure(figsize=(5, 3))
    ax_wordcloud = wordcloud_figure.add_subplot(111)
    ax_wordcloud.imshow(wordcloud, interpolation='bilinear')
    ax_wordcloud.axis('off')
    ax_wordcloud.set_title(f'Topic {topic_idx + 1}')

    # Embed the word cloud figure in the canvas
    wordcloud_canvas = FigureCanvasTkAgg(wordcloud_figure, master=canvas_wordcloud)
    wordcloud_canvas.draw()
    wordcloud_canvas.get_tk_widget().pack()

    # Set the canvas scrollable region
    canvas_wordcloud.create_window(x_pos, 0, anchor=tk.NW, window=wordcloud_canvas.get_tk_widget())
    x_pos += 210

# Create a frame for the box plot, violin plot, pie chart, donut chart, Q-Q plot, histogram, line chart, and spline chart
statistics_frame = tk.Frame(window)
statistics_frame.grid(row=3, column=0, columnspan=num_topics, padx=10, pady=10)

# Generate random data for demonstration purposes
np.random.seed(42)
data1 = np.random.normal(loc=0, scale=1, size=100)
data2 = np.random.normal(loc=1, scale=1, size=100)
data3 = np.random.normal(loc=-1, scale=1, size=100)
data4 = np.random.normal(loc=1, scale=0.5, size=100)
data5 = np.random.normal(loc=0, scale=1, size=100)
data6 = np.random.uniform(low=0, high=1, size=100)
x_data = np.linspace(0, 10, 100)
y_data = np.sin(x_data)
spline_data = np.random.normal(loc=0, scale=0.3, size=100) + y_data

# Plotting the box plot
fig_box_plot = plt.Figure(figsize=(4, 2))
ax_box_plot = fig_box_plot.add_subplot(111)
sns.boxplot(data=[data1, data2], ax=ax_box_plot)
ax_box_plot.set_xlabel('Category')
ax_box_plot.set_ylabel('Value')
ax_box_plot.set_title('Box Plot')
canvas_box_plot = FigureCanvasTkAgg(fig_box_plot, master=statistics_frame)
canvas_box_plot.draw()
canvas_box_plot.get_tk_widget().pack(side=tk.LEFT, padx=10)

# Plotting the violin plot
fig_violin_plot = plt.Figure(figsize=(4, 2))
ax_violin_plot = fig_violin_plot.add_subplot(111)
sns.violinplot(data=[data1, data2], ax=ax_violin_plot)
ax_violin_plot.set_xlabel('Category')
ax_violin_plot.set_ylabel('Value')
ax_violin_plot.set_title('Violin Plot')
canvas_violin_plot = FigureCanvasTkAgg(fig_violin_plot, master=statistics_frame)
canvas_violin_plot.draw()
canvas_violin_plot.get_tk_widget().pack(side=tk.LEFT, padx=10)

# Plotting the Kernel Density Plot
fig_kde_plot = plt.Figure(figsize=(4, 2))
ax_kde_plot = fig_kde_plot.add_subplot(111)
sns.kdeplot(data3, shade=True, ax=ax_kde_plot)
sns.kdeplot(data4, shade=True, ax=ax_kde_plot)
ax_kde_plot.set_xlabel('Value')
ax_kde_plot.set_ylabel('Density')
ax_kde_plot.set_title('Kernel Density Plot')
canvas_kde_plot = FigureCanvasTkAgg(fig_kde_plot, master=statistics_frame)
canvas_kde_plot.draw()
canvas_kde_plot.get_tk_widget().pack(side=tk.LEFT, padx=10)

# Plotting the CD plot
fig_cd_plot = plt.Figure(figsize=(4, 2))
ax_cd_plot = fig_cd_plot.add_subplot(111)
sns.kdeplot(data1, cumulative=True, ax=ax_cd_plot, label='Data 1')
sns.kdeplot(data2, cumulative=True, ax=ax_cd_plot, label='Data 2')
sns.kdeplot(data3, cumulative=True, ax=ax_cd_plot, label='Data 3')
sns.kdeplot(data4, cumulative=True, ax=ax_cd_plot, label='Data 4')
ax_cd_plot.set_xlabel('Value')
ax_cd_plot.set_ylabel('Cumulative Probability')
ax_cd_plot.set_title('CD Plot')
ax_cd_plot.legend()
canvas_cd_plot = FigureCanvasTkAgg(fig_cd_plot, master=statistics_frame)
canvas_cd_plot.draw()
canvas_cd_plot.get_tk_widget().pack(side=tk.LEFT, padx=10)
chart_frame = tk.Frame(window)
chart_frame.grid(row=4, column=0, columnspan=num_topics, padx=10, pady=10)
# Plotting the Q-Q Plot
fig_qq_plot = plt.Figure(figsize=(4, 2))
ax_qq_plot = fig_qq_plot.add_subplot(111)
stats.probplot(data5, dist="norm", plot=ax_qq_plot)
ax_qq_plot.set_xlabel('Theoretical Quantiles')
ax_qq_plot.set_ylabel('Ordered Values')
ax_qq_plot.set_title('Q-Q Plot')
canvas_qq_plot = FigureCanvasTkAgg(fig_qq_plot, master=chart_frame)
canvas_qq_plot.draw()
canvas_qq_plot.get_tk_widget().pack(side=tk.LEFT, padx=10)

# Plotting the histogram
fig_histogram = plt.Figure(figsize=(4, 2))
ax_histogram = fig_histogram.add_subplot(111)
ax_histogram.hist(data6, bins=10, color='steelblue')
ax_histogram.set_xlabel('Value')
ax_histogram.set_ylabel('Frequency')
ax_histogram.set_title('Histogram')
canvas_histogram = FigureCanvasTkAgg(fig_histogram, master=chart_frame)
canvas_histogram.draw()
canvas_histogram.get_tk_widget().pack(side=tk.LEFT, padx=10)

# Plotting the line chart
fig_line_chart = plt.Figure(figsize=(4, 2))
ax_line_chart = fig_line_chart.add_subplot(111)
ax_line_chart.plot(x_data, y_data, color='steelblue')
ax_line_chart.set_xlabel('X')
ax_line_chart.set_ylabel('Y')
ax_line_chart.set_title('Line Chart')
canvas_line_chart = FigureCanvasTkAgg(fig_line_chart, master=plot_frame)
canvas_line_chart.draw()
canvas_line_chart.get_tk_widget().pack(side=tk.LEFT, padx=10)

# Plotting the spline chart
fig_spline_chart = plt.Figure(figsize=(4, 2))
ax_spline_chart = fig_spline_chart.add_subplot(111)
ax_spline_chart.plot(x_data, spline_data, color='steelblue')
ax_spline_chart.set_xlabel('X')
ax_spline_chart.set_ylabel('Y')
ax_spline_chart.set_title('Spline Chart')
canvas_spline_chart = FigureCanvasTkAgg(fig_spline_chart, master=plot_frame)
canvas_spline_chart.draw()
canvas_spline_chart.get_tk_widget().pack(side=tk.LEFT, padx=10)

# Create a frame for the pie chart and donut chart


# Calculate the patent count per topic as a list
patent_count_per_topic = df_topic_distribution.sum().tolist()

# Plotting the pie chart
fig_pie_chart = plt.Figure(figsize=(4, 2))
ax_pie_chart = fig_pie_chart.add_subplot(111)
ax_pie_chart.pie(patent_count_per_topic, labels=topic_names, autopct='%1.1f%%', startangle=90)
ax_pie_chart.set_title('Patent Count Distribution')
canvas_pie_chart = FigureCanvasTkAgg(fig_pie_chart, master=chart_frame)
canvas_pie_chart.draw()
canvas_pie_chart.get_tk_widget().pack(side=tk.LEFT, padx=10)

# Plotting the donut chart
fig_donut_chart = plt.Figure(figsize=(4, 2))
ax_donut_chart = fig_donut_chart.add_subplot(111)
ax_donut_chart.pie(patent_count_per_topic, labels=topic_names, autopct='%1.1f%%', startangle=90, wedgeprops=dict(width=0.3))
ax_donut_chart.set_title('Patent Count Distribution (Donut)')
canvas_donut_chart = FigureCanvasTkAgg(fig_donut_chart, master=chart_frame)
canvas_donut_chart.draw()
canvas_donut_chart.get_tk_widget().pack(side=tk.RIGHT, padx=10)

window.mainloop()
