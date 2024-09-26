import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Read your CSV
df = pd.read_csv('/Users/caioteixeira/PycharmProjects/etfperformance/portfolio_data/predictions.csv')

# Group by ETF
grouped = df.groupby('Symbol')

# Create figure with make_subplots, specifying subplot types
fig = make_subplots(
    rows=len(grouped), cols=4,
    subplot_titles=[f'{s}' for s in grouped.groups.keys()],
    specs=[[
        {'type': 'domain'},  # For gauge chart
        {'type': 'xy'},  # For bar chart
        {'type': 'xy'},  # For double bar chart
        {'type': 'xy'}  # For scatter plot
    ] for _ in range(len(grouped))]
)

# Loop through each ETF
for i, (symbol, group) in enumerate(grouped):
    row = i + 1

    # Volatility Gauge Chart
    fig.add_trace(
        go.Indicator(
            mode="gauge+number+delta",
            value=group['Volatility'].iloc[0],
            title={'text': "Volatility"},
            delta={'reference': group['MA50'].iloc[0]},
            gauge={
                'axis': {'range': [0, max(group['Volatility'])]},
                'bar': {'color': "darkblue"},
                'bgcolor': "white",
                'steps': [
                    {'range': [0, group['Volatility'].iloc[0]], 'color': 'lightgray'},
                ]
            }
        ),
        row=row, col=1
    )

    # Performance Bar Chart
    fig.add_trace(
        go.Bar(
            y=['Today', '5Y', '10Y', '15Y'],
            x=[group['today'].iloc[0], group['5 years'].iloc[0], group['10 years'].iloc[0], group['15 years'].iloc[0]],
            marker_color=['blue' if x > 0 else 'red' for x in
                          [group['today'].iloc[0], group['5 years'].iloc[0], group['10 years'].iloc[0],
                           group['15 years'].iloc[0]]],
            text=[f'{x:.2f}%' for x in [group['today'].iloc[0], group['5 years'].iloc[0], group['10 years'].iloc[0],
                                        group['15 years'].iloc[0]]],
            textposition='outside'
        ),
        row=row, col=2
    )

    # Amount (with Expense Ratio in red)
    amount_bar = go.Bar(
        y=[symbol],
        x=[group['Current Price'].iloc[0] * group['Quantity'].iloc[0]],
        marker_color='lightgrey'
    )
    expense_bar = go.Bar(
        y=[symbol],
        x=[group['Expense_Ratio'].iloc[0]],
        marker_color='red'
    )
    fig.add_trace(amount_bar, row=row, col=3)
    fig.add_trace(expense_bar, row=row, col=3)

    # Line of Tendency
    fig.add_trace(
        go.Scatter(
            x=group['Date'],
            y=group['Current Price'],
            mode='lines+markers',
            line=dict(color='purple')
        ),
        row=row, col=4
    )

# Update layout for better visual organization
fig.update_layout(height=300 * len(grouped), title_text="ETF Performance Overview", showlegend=False)
fig.for_each_annotation(lambda a: a.update(textangle=0))

# Show or save the figure
fig.show()
# or fig.write_html("etf_overview.html")  # to save as an interactive HTML file