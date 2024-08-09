# views.py
from django.shortcuts import render, redirect
from django.http import HttpResponseBadRequest,HttpResponse
from django.core.files.storage import FileSystemStorage
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import base64
from io import  BytesIO,StringIO
from matplotlib.backends.backend_agg import FigureCanvasAgg
import json
from urllib.request import urlopen
from matplotlib.patches import Circle, Rectangle
import mpld3
from urllib.request import urlopen
import matplotlib
matplotlib.use('Agg')

def about(request):
    return render(request, 'about.html')

def report(request):
    return render(request, 'report.html')




def upload_csv(request):
    if request.method == 'POST':
        csv_file = request.FILES.get('csv_file')
        if not csv_file:
            return HttpResponseBadRequest("No file uploaded.")
        if csv_file.name.endswith('.csv'):
            fs = FileSystemStorage()
            filename = fs.save(csv_file.name, csv_file)
            file_url = fs.url(filename)
            
            # Read the CSV file
            df = pd.read_csv(fs.path(filename))

            # Save the DataFrame to the session
            request.session['csv_data'] = df.to_json()

            # Redirect to the data view
            return redirect('data')
        else:
            return HttpResponseBadRequest("File is not a CSV.")
    return render(request, 'index.html')

def data(request):
    csv_data = request.session.get('csv_data')
    if csv_data:
        df = pd.read_json(csv_data)
        columns = df.columns.tolist()
        rows = df.head(10).values.tolist()
    else:
        columns = []
        rows = []

    return render(request, 'data.html', {'columns': columns, 'rows': rows})

def convert_object_to_numeric(df):
    for column in df.columns:
        df[column] = pd.to_numeric(df[column], errors='ignore', downcast='float')
    return df

def round_values(dictionary, decimals=2):
    for key, value in dictionary.items():
        if isinstance(value, dict):
            dictionary[key] = round_values(value, decimals)
        elif isinstance(value, float):
            dictionary[key] = round(value, decimals)
    return dictionary

def get_statistics(df):
    df = convert_object_to_numeric(df)
    num_df = df.select_dtypes(include=['number', 'float'])
    cat_df = df.select_dtypes(include='object')
    shape = df.shape
    cat_df_categorical = df.select_dtypes(include='category')
    cat_df_combined = pd.concat([cat_df, cat_df_categorical], axis=1)
    
    num_stats = num_df.describe().T if not num_df.empty else pd.DataFrame()
    cat_stats = cat_df_combined.describe().T if not cat_df_combined.empty else pd.DataFrame()

    return {
        'num_stats': num_stats.to_html(classes='table table-striped', border=0) if not num_stats.empty else '<p>No numerical data available.</p>',
        'cat_stats': cat_stats.to_html(classes='table table-striped', border=0) if not cat_stats.empty else '<p>No categorical data available.</p>',
    }

def statistics_original(request):
    csv_data = request.session.get('csv_data')
    if csv_data:
        df = pd.read_json(csv_data)
        context = get_statistics(df)
    else:
        context = {
            'num_stats': '<p>No numerical data available.</p>',
            'cat_stats': '<p>No categorical data available.</p>',
        }
    return render(request, 'statistics_original.html', context)

def statistics_cleaned(request):
    csv_data = request.session.get('processed_df')
    if csv_data:
        df = pd.read_json(csv_data)
        context = get_statistics(df)
    else:
        context = {
            'num_stats': '<p>No numerical data available.</p>',
            'cat_stats': '<p>No categorical data available.</p>',
        }
    return render(request, 'statistics_original.html', context)

def handle_missing_values(request):
    if request.method == 'POST':
        action = request.POST.get('action')
        csv_data = request.session.get('csv_data')
        if csv_data:
            df = pd.read_json(csv_data)
            if action == 'drop_column':
                df = df.dropna(axis=1, how='any')
            elif action == 'drop_row':
                df = df.dropna(axis=0, how='any')
            elif action == 'fill_mean':
                df = df.fillna(df.mean())
            elif action == 'fill_median':
                df = df.fillna(df.median())
            elif action == 'fill_mode':
                df = df.fillna(df.mode().iloc[0])
            elif action == 'fill_forward':
                df = df.fillna(method='ffill')
            elif action == 'fill_backward':
                df = df.fillna(method='bfill')

            # Save the cleaned DataFrame to the session
            request.session['processed_df'] = df.to_json()

            # Recompute missing values for the cleaned DataFrame
            cleaned_missing_values = df.isnull().sum()
            cleaned_percent_missing = (df.isnull().mean() * 100).round(2)
            cleaned_missing_values_df = pd.DataFrame({
                'Column': df.columns,
                'Missing Values': cleaned_missing_values,
                'Percentage': cleaned_percent_missing
            })
            cleaned_missing_values_table = cleaned_missing_values_df.to_html(classes='table table-striped', border=0)

            # Redirect to the same page to show updated statistics
            return redirect('handle_missing_values')

    # Retrieve and display missing values for the original DataFrame
    csv_data = request.session.get('csv_data')
    if csv_data:
        df = pd.read_json(csv_data)
        num_missing = df.isnull().sum()
        percent_missing = (df.isnull().mean() * 100).round(2)
        missing_values_df = pd.DataFrame({
            'Column': df.columns,
            'Missing Values': num_missing,
            'Percentage': percent_missing
        })
        missing_values_table = missing_values_df.to_html(classes='table table-striped', border=0)

        # Retrieve the cleaned DataFrame from the session and compute statistics
        cleaned_df_json = request.session.get('processed_df')
        if cleaned_df_json:
            cleaned_df = pd.read_json(cleaned_df_json)
            cleaned_num_missing = cleaned_df.isnull().sum()
            cleaned_percent_missing = (cleaned_df.isnull().mean() * 100).round(2)
            cleaned_missing_values_df = pd.DataFrame({
                'Column': cleaned_df.columns,
                'Missing Values': cleaned_num_missing,
                'Percentage': cleaned_percent_missing
            })
            cleaned_missing_values_table = cleaned_missing_values_df.to_html(classes='table table-striped', border=0)
        else:
            cleaned_missing_values_table = '<p>No cleaned data available.</p>'
    else:
        missing_values_table = '<p>No data available.</p>'
        cleaned_missing_values_table = '<p>No data available.</p>'

    context = {
        'missing_values_table': missing_values_table,
        'cleaned_missing_values_table': cleaned_missing_values_table
    }
    return render(request, 'handle_missing_values.html', context)


def show_plot_page(request):
    return render(request, 'plot.html')

def generate_plot(request):
        csv_data = request.session.get('csv_data')
        if csv_data:
            df = pd.read_json(csv_data)

            numerical_columns = df.select_dtypes(include=['float','int']).columns
            num_cols = len(numerical_columns)
            fig, axes = plt.subplots(nrows=(num_cols + 1) // 2, ncols=2, figsize=(20,5* ((num_cols +1) // 2)))
            axes = axes.flatten()
            for i, col in enumerate(numerical_columns):
                colors = generate_random_color()
                sns.histplot(df[col],ax=axes[i], kde=True,color=colors)
                axes[i].set_title(f'Histogram of {col}')

            for j in range(i + 1, len(axes)):
                fig.delaxes(axes[j])
            
            
    # Save the plot to a BytesIO object
            buffer = BytesIO()
            plt.savefig(buffer, format='png')
            plt.close()
            buffer.seek(0)

            image_data = base64.b64encode(buffer.getvalue()).decode('utf-8')

            return render(request, 'plot.html',{'image_data': image_data})
        else:
            return HttpResponse("No CSV data found in session.", status=400)

def generate_kdeplot(request):
        csv_data = request.session.get('csv_data')
        if csv_data:
            df = pd.read_json(csv_data)

            numerical_columns = df.select_dtypes(include=['float','int']).columns
            num_cols = len(numerical_columns)
            fig, axes = plt.subplots(nrows=(num_cols + 1) // 2, ncols=2, figsize=(20,5* ((num_cols +1) // 2)))
            axes = axes.flatten()
            for i, col in enumerate(numerical_columns):
                colors = generate_random_color()
                sns.kdeplot(df[col],ax=axes[i],color=colors,shade=True)
                axes[i].set_title(f'Kdeplot of {col}')

            for j in range(i + 1, len(axes)):
                fig.delaxes(axes[j])
            
            
    # Save the plot to a BytesIO object
            buffer = BytesIO()
            plt.savefig(buffer, format='png')
            plt.close()
            buffer.seek(0)

            image_data = base64.b64encode(buffer.getvalue()).decode('utf-8')

            return render(request, 'plot.html',{'image_data': image_data})
        else:
            return HttpResponse("No CSV data found in session.", status=400)


def generate_scatterplot(request):
    csv_data = request.session.get('csv_data')
    if csv_data:
        # Convert JSON data to DataFrame
        df = pd.read_json(csv_data)

        # Identify numerical columns
        numerical_columns = df.select_dtypes(include=['float64', 'int64']).columns
        num_cols = len(numerical_columns)
        
        # Create subplots
        fig, axes = plt.subplots(nrows=(num_cols * (num_cols - 1) // 2 + 1) // 2, ncols=3, figsize=(20, 5 * ((num_cols * (num_cols - 1) // 2 + 1) // 2)))
        axes = axes.flatten()

        # Generate scatterplots
        plot_index = 0
        for i in range(num_cols):
            for j in range(i + 1, num_cols):
                col1 = numerical_columns[i]
                col2 = numerical_columns[j]
                sns.scatterplot(data=df, x=col1, y=col2, ax=axes[plot_index])
                axes[plot_index].set_title(f'Scatterplot: {col1} vs {col2}')
                plot_index += 1

        # Remove unused subplots
        for j in range(plot_index, len(axes)):
            fig.delaxes(axes[j])

        # Save the plot to a BytesIO object
        buffer = BytesIO()
        plt.tight_layout()
        plt.savefig(buffer, format='png')
        plt.close()
        buffer.seek(0)

        image_data = base64.b64encode(buffer.getvalue()).decode('utf-8')

        return render(request, 'plot.html',{'image_data': image_data})
    else:
        return HttpResponse("No CSV data found in session.", status=400)
 
def generate_barplot(request):
    csv_data = request.session.get('csv_data')
    if csv_data:
        # Convert JSON data to DataFrame
        df = pd.read_json(csv_data)

        # Identify numerical and categorical columns
        numerical_columns = df.select_dtypes(include=['float64', 'int64']).columns
        categorical_columns = df.select_dtypes(include=['object']).columns
        print(categorical_columns)
        # Check if there are categorical columns
        if categorical_columns.empty:
            return HttpResponse("No categorical columns available for bar plots.", status=400)

        # Create subplots for bar plots
        num_plots = len(numerical_columns) * len(categorical_columns)
        num_rows = (num_plots + 2) // 3
        fig, axes = plt.subplots(nrows=num_rows, ncols=3, figsize=(20, 5 * num_rows))
        axes = axes.flatten()

        # Generate bar plots
        plot_index = 0
        for num_col in numerical_columns:
            for cat_col in categorical_columns:
                random_color = generate_random_color()

                sns.barplot(data=df, x=cat_col, y=num_col, ax=axes[plot_index],color=random_color)
                axes[plot_index].set_title(f'Barplot: {num_col} by {cat_col}')
                plot_index += 1

        # Remove unused subplots
        for j in range(plot_index, len(axes)):
            fig.delaxes(axes[j])

        # Save the plot to a BytesIO object
        buffer = BytesIO()
        plt.tight_layout()
        plt.savefig(buffer, format='png')
        plt.close()
        buffer.seek(0)

        image_data = base64.b64encode(buffer.getvalue()).decode('utf-8')

        return render(request, 'plot.html',{'image_data': image_data})
    else:
        return HttpResponse("No CSV data found in session.", status=400)
    
def generate_random_colors(n):
    """Generates a list of n random colors."""
    return [(random.random(), random.random(), random.random()) for _ in range(n)]

def generate_piechart(request):
    csv_data = request.session.get('csv_data')
    if csv_data:
        # Convert JSON data to DataFrame
        json_buffer = StringIO(csv_data)
        df = pd.read_json(json_buffer)

        # Identify categorical columns
        categorical_columns = df.select_dtypes(include=['object']).columns

        # Check if there are categorical columns
        if categorical_columns.empty:
            return HttpResponse("No categorical columns available for pie charts.", status=400)

        # Create subplots for pie charts
        num_plots = len(categorical_columns)
        num_rows = (num_plots + 2) // 3
        fig, axes = plt.subplots(nrows=num_rows, ncols=3, figsize=(20, 5 * num_rows))
        axes = axes.flatten()

        # Generate pie charts
        for i, cat_col in enumerate(categorical_columns):
            data_counts = df[cat_col].value_counts()
            random_color = generate_random_colors(num_plots)

            axes[i].pie(data_counts, labels=data_counts.index, autopct='%1.1f%%', startangle=140,colors=random_color)
            axes[i].set_title(f'Pie Chart: {cat_col}')

        # Remove unused subplots
        for j in range(i + 1, len(axes)):
            fig.delaxes(axes[j])

        # Save the plot to a BytesIO object
        buffer = BytesIO()
        plt.tight_layout()
        plt.savefig(buffer, format='png')
        plt.close()
        buffer.seek(0)
        image_data = base64.b64encode(buffer.getvalue()).decode('utf-8')

        return render(request, 'plot.html',{'image_data': image_data})
    else:
        return HttpResponse("No CSV data found in session.", status=400)

import random     
def generate_random_color():
    """Generates a random color in RGB format."""
    return (random.random(), random.random(), random.random())

def generate_boxplot(request):
    csv_data = request.session.get('csv_data')
    if csv_data:
        df = pd.read_json(csv_data)

        numerical_columns = df.select_dtypes(include=['float','int']).columns
        num_cols = len(numerical_columns)
        fig, axes = plt.subplots(nrows=(num_cols + 1) // 2, ncols=2, figsize=(20,5* ((num_cols +1) // 2)))
        axes = axes.flatten()
        for i, col in enumerate(numerical_columns):
            random_color = generate_random_color()
            sns.boxplot(x=df[col],ax=axes[i], color=random_color)
            axes[i].set_title(f'Boxplot of {col}')

        for j in range(i + 1, len(axes)):
            fig.delaxes(axes[j])
        
        
# Save the plot to a BytesIO object
        buffer = BytesIO()
        plt.savefig(buffer, format='png')
        plt.close()
        buffer.seek(0)

        image_data = base64.b64encode(buffer.getvalue()).decode('utf-8')

        return render(request, 'plot.html',{'image_data': image_data})
    else:
        return HttpResponse("No CSV data found in session.", status=400)
 
def generate_random_palette(n_colors):
    """Generates a random color palette with n_colors colors."""
    return sns.color_palette([(random.random(), random.random(), random.random()) for _ in range(n_colors)])


def generate_heatmap(request):
    csv_data = request.session.get('csv_data')
    if csv_data:
        df = pd.read_json(csv_data)

        numerical_columns = df.select_dtypes(include=['float','int']).columns
        correlation = df[numerical_columns].corr()
        
        random_color = generate_random_palette(n_colors=10)

        sns.heatmap(correlation,annot=True,cmap=random_color)
        plt.title(f'Correlational Plot of Data')


        
# Save the plot to a BytesIO object
        buffer = BytesIO()
        plt.savefig(buffer, format='png')
        plt.close()
        buffer.seek(0)

        image_data = base64.b64encode(buffer.getvalue()).decode('utf-8')

        return render(request, 'plot.html',{'image_data': image_data})
    else:
        return HttpResponse("No CSV data found in session.", status=400)
 

def generate_lineplot(request):
    csv_data = request.session.get('csv_data')
    if csv_data:
        df = pd.read_json(csv_data)
        
        numerical_columns = df.select_dtypes(include=['float', 'int']).columns
        num_cols = len(numerical_columns)
        
        # Create subplots
        fig, axes = plt.subplots(nrows=(num_cols + 1) // 2, ncols=2, figsize=(15, 5 * ((num_cols + 1) // 2)))
        axes = axes.flatten()

        # Generate a line plot for each numerical column
        for i, col in enumerate(numerical_columns):
            random_color = generate_random_color()

            axes[i].plot(df.index, df[col], label=col,color=random_color)
            axes[i].set_title(f'Line Plot: {col}')
            axes[i].set_xlabel('Index')
            axes[i].set_ylabel('Value')
            axes[i].legend()

        # Remove unused subplots
        for j in range(len(numerical_columns), len(axes)):
            fig.delaxes(axes[j])

        plt.tight_layout()

        buffer = BytesIO()
        plt.savefig(buffer, format='png')
        plt.close()
        buffer.seek(0)

        image_data = base64.b64encode(buffer.getvalue()).decode('utf-8')
        return render(request, 'plot.html', {'image_data': image_data})
    else:
        return HttpResponse("No CSV data found in session.", status=400)
    


def get_stat(request):
    csv_data = request.session.get('csv_data')
    
    if not csv_data:
        return render(request, 'error.html', {'message': 'No data available. Please upload a CSV file.'})

    df = pd.read_json(csv_data)

    # Initialize statistics dictionaries
    stats = {
        'num_stats': None,
        'cat_stats': None,
        'variance': None,
        'iqr': None,
        'value_counts': {},
        'percentages': [],
        'shape': df.shape,
        'size': df.size,
        'missing_values': df.isnull().sum().to_dict(),
        'duplicate_count': df.duplicated().sum()
    }

    try:
        num_df = df.select_dtypes(include=['number'])
        cat_df = df.select_dtypes(include=['object'])

        if not num_df.empty:
            stats['num_stats'] = num_df.describe().T.to_dict()
            stats['variance'] = num_df.var().to_dict()
            stats['iqr'] = (num_df.quantile(0.75) - num_df.quantile(0.25)).to_dict()

        if not cat_df.empty:
            stats['cat_stats'] = cat_df.describe(include=['object']).T.to_dict()
            for col in cat_df.columns:
                value_counts = cat_df[col].value_counts().to_dict()
                percentages = (cat_df[col].value_counts(normalize=True) * 100).to_dict()

                stats['value_counts'][col] = value_counts
                for category, percent in percentages.items():
                    stats['percentages'].append({
                        'column': col,
                        'category': category,
                        'percentage': percent
                    })

    except Exception as e:
        print(f"Error calculating statistics: {e}")

    return render(request, 'statistics.html', stats)



def clean_stat(request):
    new_data = request.session.get('processed_df')
    
    if not new_data:
        return render(request, 'error.html', {'message': 'No data available. Please upload a CSV file.'})

    df = pd.read_json(new_data)

    # Initialize statistics dictionaries
    stats = {
        'num_stats': None,
        'cat_stats': None,
        'variance': None,
        'iqr': None,
        'value_counts': {},
        'percentages': [],
        'shape': df.shape,
        'size': df.size,
        'missing_values': df.isnull().sum().to_dict(),
        'duplicate_count': df.duplicated().sum()
    }

    try:
        num_df = df.select_dtypes(include=['number'])
        cat_df = df.select_dtypes(include=['object'])

        if not num_df.empty:
            stats['num_stats'] = num_df.describe().T.to_dict()
            stats['variance'] = num_df.var().to_dict()
            stats['iqr'] = (num_df.quantile(0.75) - num_df.quantile(0.25)).to_dict()

        if not cat_df.empty:
            stats['cat_stats'] = cat_df.describe(include=['object']).T.to_dict()
            for col in cat_df.columns:
                value_counts = cat_df[col].value_counts().to_dict()
                percentages = (cat_df[col].value_counts(normalize=True) * 100).to_dict()

                stats['value_counts'][col] = value_counts
                for category, percent in percentages.items():
                    stats['percentages'].append({
                        'column': col,
                        'category': category,
                        'percentage': percent
                    })

    except Exception as e:
        print(f"Error calculating statistics: {e}")

    return render(request, 'statistics.html', stats)

def show_plot(request):
    if request.method == 'POST':
        action = request.POST.get('action')
        csv_data = request.session.get('csv_data')
        if csv_data:
            df = pd.read_json(csv_data)
            
            if action == 'scatterplot':
                return redirect('generate_scatterplot')

            elif action == 'barplot':
                return redirect('generate_barplot')

            # Add more plot types as needed
            elif action == 'histogram':
                return redirect('generate_plot')

            elif action == 'piechart':
                return redirect('generate_piechart')
            
            elif action == 'boxplot':
                return redirect('generate_boxplot')
            
            elif action == 'heatmap':
                return redirect('generate_heatmap')

            elif action == 'kdeplot':
                return redirect('generate_kdeplot')
                   
            elif action == 'lineplot':
                return redirect('generate_lineplot')

            # Add any additional plot types here
            # ...

    return render(request, 'plot.html')



def custom_plot(request):
    if request.method == 'POST':
        action = request.POST.get('action')
        column1 = request.POST.get('column1')  # Get the first selected column
        column2 = request.POST.get('column2')  # Get the second selected column
        csv_data = request.session.get('csv_data')

        if csv_data:
            df = pd.read_json(csv_data)
            buffer = BytesIO()

            if action == 'scatterplot' and column1 and column2:
                plt.figure(figsize=(10, 6))
                sns.scatterplot(data=df, x=column1, y=column2)
                plt.title(f'Scatterplot of {column1} vs {column2}')
                plt.xlabel(column1)
                plt.ylabel(column2)
                plt.savefig(buffer, format='png')
                plt.close()
                buffer.seek(0)
                image_data = base64.b64encode(buffer.getvalue()).decode('utf-8')
                return render(request, 'show_plot.html', {'image_data': image_data})

            elif action == 'barplot' and column1 and column2:
                plt.figure(figsize=(10, 6))
                sns.barplot(data=df, x=column1, y=column2)
                plt.title(f'Barplot of {column1} vs {column2}')
                plt.xlabel(column1)
                plt.ylabel(column2)
                plt.savefig(buffer, format='png')
                plt.close()
                buffer.seek(0)
                image_data = base64.b64encode(buffer.getvalue()).decode('utf-8')
                return render(request, 'show_plot.html', {'image_data': image_data})

            elif action == 'histogram' and column1:
                plt.figure(figsize=(10, 6))
                sns.histplot(df[column1], bins=30, kde=True)
                plt.title(f'Histogram of {column1}')
                plt.xlabel(column1)
                plt.ylabel('Frequency')
                plt.savefig(buffer, format='png')
                plt.close()
                buffer.seek(0)
                image_data = base64.b64encode(buffer.getvalue()).decode('utf-8')
                return render(request, 'show_plot.html', {'image_data': image_data})

            elif action == 'piechart' and column1:
                plt.figure(figsize=(8, 8))
                df[column1].value_counts().plot.pie(autopct='%1.1f%%', colors=sns.color_palette("pastel"))
                plt.title(f'Pie Chart of {column1}')
                plt.ylabel('')
                plt.savefig(buffer, format='png')
                plt.close()
                buffer.seek(0)
                image_data = base64.b64encode(buffer.getvalue()).decode('utf-8')
                return render(request, 'show_plot.html', {'image_data': image_data})

            elif action == 'boxplot' and column1:
                plt.figure(figsize=(10, 6))
                sns.boxplot(data=df, y=column1)
                plt.title(f'Boxplot of {column1}')
                plt.ylabel(column1)
                plt.savefig(buffer, format='png')
                plt.close()
                buffer.seek(0)
                image_data = base64.b64encode(buffer.getvalue()).decode('utf-8')
                return render(request, 'show_plot.html', {'image_data': image_data})

            elif action == 'heatmap' and column1 and column2:
                plt.figure(figsize=(10, 8))
                corr = df[[column1, column2]].corr()
                sns.heatmap(corr, annot=True, cmap='coolwarm')
                plt.title(f'Heatmap of {column1} and {column2}')
                plt.savefig(buffer, format='png')
                plt.close()
                buffer.seek(0)
                image_data = base64.b64encode(buffer.getvalue()).decode('utf-8')
                return render(request, 'show_plot.html', {'image_data': image_data})

            elif action == 'lineplot' and column1 and column2:
                plt.figure(figsize=(10, 6))
                sns.lineplot(data=df, x=column1, y=column2)
                plt.title(f'Line Plot of {column1} vs {column2}')
                plt.xlabel(column1)
                plt.ylabel(column2)
                plt.savefig(buffer, format='png')
                plt.close()
                buffer.seek(0)
                image_data = base64.b64encode(buffer.getvalue()).decode('utf-8')
                return render(request, 'show_plot.html', {'image_data': image_data})

            # Handle any additional plot types here
            # ...

    # If not POST request or missing data, render plot selection page
    csv_data = request.session.get('csv_data')
    if csv_data:
        df = pd.read_json(csv_data)
        columns = df.columns.tolist()
    else:
        columns = []

    return render(request, 'show_plot.html', {'columns': columns})