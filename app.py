# =============================================
# Advanced Salary Prediction ML Application
# Enhanced Features:
#   ✅ Modern Tab-Based Interface
#   ✅ Multiple ML Models (Linear, Polynomial, Ridge, Lasso)
#   ✅ Comprehensive Model Metrics Dashboard
#   ✅ Interactive Multi-Plot Visualization
#   ✅ Cross-Validation & Feature Analysis
#   ✅ Model Comparison Tools
#   ✅ Export Predictions & Reports
# =============================================

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import seaborn as sns
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from datetime import datetime

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.facecolor'] = 'white'


class SalaryPredictionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("🎯 Advanced Salary Prediction & ML Analytics")
        self.root.geometry("1200x800")
        self.root.configure(bg='#f0f0f0')

        # Variables
        self.df = None
        self.models = {}
        self.current_model = None
        self.model_name = tk.StringVar(value="Linear Regression")
        self.history = []
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

        self.setup_ui()

    def setup_ui(self):
        """Setup the main UI with tabs"""
        # Title Frame
        title_frame = tk.Frame(self.root, bg='#2c3e50', height=60)
        title_frame.pack(fill='x', side='top')
        title_frame.pack_propagate(False)

        tk.Label(title_frame, text="🎯 Advanced Salary Prediction & ML Analytics",
                 font=("Arial", 18, "bold"), fg='white', bg='#2c3e50').pack(pady=15)

        # Create Notebook (Tabs)
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill='both', expand=True, padx=10, pady=10)

        # Tab 1: Data & Training
        self.tab1 = ttk.Frame(self.notebook)
        self.notebook.add(self.tab1, text="📊 Data & Training")
        self.create_data_tab()

        # Tab 2: Visualization
        self.tab2 = ttk.Frame(self.notebook)
        self.notebook.add(self.tab2, text="📈 Visualizations")
        self.create_visualization_tab()

        # Tab 3: Prediction
        self.tab3 = ttk.Frame(self.notebook)
        self.notebook.add(self.tab3, text="🔮 Prediction")
        self.create_prediction_tab()

        # Tab 4: Model Comparison
        self.tab4 = ttk.Frame(self.notebook)
        self.notebook.add(self.tab4, text="⚖️ Model Comparison")
        self.create_comparison_tab()

        # Status Bar
        self.status_bar = tk.Label(self.root, text="Ready | No data loaded",
                                   bd=1, relief=tk.SUNKEN, anchor=tk.W,
                                   bg='#ecf0f1', font=("Arial", 9))
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)

    def create_data_tab(self):
        """Create data loading and training tab"""
        # Left Panel - Controls
        left_panel = tk.Frame(self.tab1, bg='white', relief=tk.RAISED, bd=2)
        left_panel.pack(side='left', fill='both', padx=10, pady=10, expand=False)

        tk.Label(left_panel, text="Data Management", font=("Arial", 14, "bold"),
                 bg='white').pack(pady=10)

        # Load Data Button
        btn_load = tk.Button(left_panel, text="📂 Load CSV Dataset",
                             command=self.load_data, width=25, height=2,
                             bg='#3498db', fg='white', font=("Arial", 10, "bold"),
                             cursor='hand2')
        btn_load.pack(pady=10, padx=20)

        # Model Selection
        tk.Label(left_panel, text="Select ML Model:", font=("Arial", 11, "bold"),
                 bg='white').pack(pady=(20, 5))

        models = ["Linear Regression", "Polynomial (Degree 2)",
                  "Polynomial (Degree 3)", "Ridge Regression", "Lasso Regression"]

        for model in models:
            rb = tk.Radiobutton(left_panel, text=model, variable=self.model_name,
                                value=model, bg='white', font=("Arial", 10),
                                command=self.train_selected_model)
            rb.pack(anchor='w', padx=30, pady=3)

        # Train Button
        btn_train = tk.Button(left_panel, text="🚀 Train Model",
                              command=self.train_selected_model, width=25, height=2,
                              bg='#2ecc71', fg='white', font=("Arial", 10, "bold"),
                              cursor='hand2')
        btn_train.pack(pady=20, padx=20)

        # Export Button
        btn_export = tk.Button(left_panel, text="💾 Export Report",
                               command=self.export_report, width=25,
                               bg='#95a5a6', fg='white', font=("Arial", 9),
                               cursor='hand2')
        btn_export.pack(pady=5, padx=20)

        # Right Panel - Metrics Display
        right_panel = tk.Frame(self.tab1, bg='white', relief=tk.RAISED, bd=2)
        right_panel.pack(side='right', fill='both', padx=10, pady=10, expand=True)

        tk.Label(right_panel, text="Model Performance Metrics",
                 font=("Arial", 14, "bold"), bg='white').pack(pady=10)

        # Metrics Frame
        metrics_frame = tk.Frame(right_panel, bg='#ecf0f1', relief=tk.SUNKEN, bd=2)
        metrics_frame.pack(fill='both', padx=20, pady=10, expand=True)

        self.metrics_text = tk.Text(metrics_frame, height=20, width=60,
                                    font=("Courier", 10), bg='#ecf0f1',
                                    relief=tk.FLAT, padx=10, pady=10)
        self.metrics_text.pack(fill='both', expand=True)
        self.metrics_text.insert('1.0', "📊 Load data and train a model to see metrics...")

        # Data Info Frame
        info_frame = tk.Frame(right_panel, bg='white')
        info_frame.pack(fill='x', padx=20, pady=10)

        tk.Label(info_frame, text="Dataset Information:", font=("Arial", 11, "bold"),
                 bg='white').pack(anchor='w')

        self.data_info_label = tk.Label(info_frame, text="No data loaded",
                                        font=("Arial", 9), bg='white',
                                        justify='left', anchor='w')
        self.data_info_label.pack(anchor='w', pady=5)

    def create_visualization_tab(self):
        """Create visualization tab with multiple plots"""
        # Control Panel
        control_frame = tk.Frame(self.tab2, bg='white', height=60)
        control_frame.pack(fill='x', padx=10, pady=5)
        control_frame.pack_propagate(False)

        tk.Label(control_frame, text="Select Visualization:",
                 font=("Arial", 11, "bold"), bg='white').pack(side='left', padx=10)

        btn_plot1 = tk.Button(control_frame, text="📊 Regression Plot",
                              command=self.plot_regression, bg='#3498db',
                              fg='white', cursor='hand2')
        btn_plot1.pack(side='left', padx=5)

        btn_plot2 = tk.Button(control_frame, text="📉 Residuals Plot",
                              command=self.plot_residuals, bg='#9b59b6',
                              fg='white', cursor='hand2')
        btn_plot2.pack(side='left', padx=5)

        btn_plot3 = tk.Button(control_frame, text="📈 Distribution",
                              command=self.plot_distribution, bg='#e74c3c',
                              fg='white', cursor='hand2')
        btn_plot3.pack(side='left', padx=5)

        btn_plot4 = tk.Button(control_frame, text="🎯 Actual vs Predicted",
                              command=self.plot_actual_vs_predicted, bg='#1abc9c',
                              fg='white', cursor='hand2')
        btn_plot4.pack(side='left', padx=5)

        btn_plot5 = tk.Button(control_frame, text="📊 All Plots",
                              command=self.plot_all, bg='#34495e',
                              fg='white', cursor='hand2')
        btn_plot5.pack(side='left', padx=5)

        # Plot Canvas
        self.plot_frame = tk.Frame(self.tab2, bg='white')
        self.plot_frame.pack(fill='both', expand=True, padx=10, pady=10)

    def create_prediction_tab(self):
        """Create prediction tab"""
        # Input Frame
        input_frame = tk.Frame(self.tab3, bg='white', relief=tk.RAISED, bd=2)
        input_frame.pack(fill='x', padx=20, pady=20)

        tk.Label(input_frame, text="🔮 Salary Prediction",
                 font=("Arial", 16, "bold"), bg='white').pack(pady=15)

        entry_frame = tk.Frame(input_frame, bg='white')
        entry_frame.pack(pady=10)

        tk.Label(entry_frame, text="Years of Experience:",
                 font=("Arial", 12), bg='white').pack(side='left', padx=10)

        self.entry_exp = tk.Entry(entry_frame, width=15, font=("Arial", 14),
                                  relief=tk.SOLID, bd=2)
        self.entry_exp.pack(side='left', padx=10)

        btn_predict = tk.Button(entry_frame, text="🔮 Predict",
                                command=self.predict_salary, bg='#2ecc71',
                                fg='white', font=("Arial", 12, "bold"),
                                cursor='hand2', width=12)
        btn_predict.pack(side='left', padx=10)

        # Result Frame
        self.result_frame = tk.Frame(input_frame, bg='#ecf0f1', relief=tk.SUNKEN, bd=3)
        self.result_frame.pack(pady=20, padx=20, fill='x')

        self.lbl_result = tk.Label(self.result_frame, text="Enter experience and click Predict",
                                   font=("Arial", 14, "bold"), fg='#2c3e50',
                                   bg='#ecf0f1', pady=20)
        self.lbl_result.pack()

        # History Frame
        history_frame = tk.Frame(self.tab3, bg='white', relief=tk.RAISED, bd=2)
        history_frame.pack(fill='both', expand=True, padx=20, pady=10)

        tk.Label(history_frame, text="📜 Prediction History",
                 font=("Arial", 14, "bold"), bg='white').pack(pady=10)

        # History Table
        columns = ('Experience', 'Predicted Salary', 'Model', 'Timestamp')
        self.history_tree = ttk.Treeview(history_frame, columns=columns,
                                         show='headings', height=15)

        for col in columns:
            self.history_tree.heading(col, text=col)
            self.history_tree.column(col, width=150, anchor='center')

        scrollbar = ttk.Scrollbar(history_frame, orient='vertical',
                                  command=self.history_tree.yview)
        self.history_tree.configure(yscrollcommand=scrollbar.set)

        self.history_tree.pack(side='left', fill='both', expand=True, padx=10, pady=10)
        scrollbar.pack(side='right', fill='y', pady=10)

    def create_comparison_tab(self):
        """Create model comparison tab"""
        tk.Label(self.tab4, text="⚖️ Model Performance Comparison",
                 font=("Arial", 16, "bold"), bg='white').pack(pady=15)

        btn_compare = tk.Button(self.tab4, text="🔄 Compare All Models",
                                command=self.compare_models, bg='#e67e22',
                                fg='white', font=("Arial", 12, "bold"),
                                cursor='hand2', width=20, height=2)
        btn_compare.pack(pady=10)

        self.comparison_frame = tk.Frame(self.tab4, bg='white')
        self.comparison_frame.pack(fill='both', expand=True, padx=20, pady=10)

    def load_data(self):
        """Load CSV dataset"""
        file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
        if not file_path:
            return

        try:
            self.df = pd.read_csv(file_path)

            if 'YearsExperience' not in self.df.columns or 'Salary' not in self.df.columns:
                messagebox.showerror("Error",
                                     "CSV must contain 'YearsExperience' and 'Salary' columns!")
                return

            # Update data info
            info = f"✅ Dataset loaded successfully!\n"
            info += f"📊 Total Records: {len(self.df)}\n"
            info += f"📈 Experience Range: {self.df['YearsExperience'].min():.1f} - {self.df['YearsExperience'].max():.1f} years\n"
            info += f"💰 Salary Range: ${self.df['Salary'].min():,.2f} - ${self.df['Salary'].max():,.2f}"

            self.data_info_label.config(text=info)
            self.status_bar.config(text=f"✅ Data loaded | {len(self.df)} records")

            messagebox.showinfo("Success", f"Dataset loaded successfully!\n{len(self.df)} records found.")

        except Exception as e:
            messagebox.showerror("Error", f"Failed to load data:\n{e}")

    def train_selected_model(self):
        """Train the selected model"""
        if self.df is None:
            messagebox.showwarning("Warning", "Please load data first!")
            return

        try:
            X = self.df[['YearsExperience']].values
            y = self.df['Salary'].values

            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                X, y, test_size=0.2, random_state=42)

            model_type = self.model_name.get()

            if model_type == "Linear Regression":
                self.current_model = LinearRegression()
                X_train_transformed = self.X_train
                X_test_transformed = self.X_test

            elif "Polynomial" in model_type:
                degree = 2 if "Degree 2" in model_type else 3
                poly = PolynomialFeatures(degree=degree)
                X_train_transformed = poly.fit_transform(self.X_train)
                X_test_transformed = poly.transform(self.X_test)
                self.current_model = LinearRegression()
                self.models[f'poly_{degree}'] = poly

            elif model_type == "Ridge Regression":
                self.current_model = Ridge(alpha=1.0)
                X_train_transformed = self.X_train
                X_test_transformed = self.X_test

            elif model_type == "Lasso Regression":
                self.current_model = Lasso(alpha=1.0)
                X_train_transformed = self.X_train
                X_test_transformed = self.X_test

            # Train model
            self.current_model.fit(X_train_transformed, self.y_train)

            # Predictions
            y_train_pred = self.current_model.predict(X_train_transformed)
            y_test_pred = self.current_model.predict(X_test_transformed)

            # Metrics
            train_r2 = r2_score(self.y_train, y_train_pred)
            test_r2 = r2_score(self.y_test, y_test_pred)
            train_mae = mean_absolute_error(self.y_train, y_train_pred)
            test_mae = mean_absolute_error(self.y_test, y_test_pred)
            train_rmse = np.sqrt(mean_squared_error(self.y_train, y_train_pred))
            test_rmse = np.sqrt(mean_squared_error(self.y_test, y_test_pred))

            # Cross-validation
            cv_scores = cross_val_score(self.current_model, X_train_transformed,
                                        self.y_train, cv=5, scoring='r2')

            # Display metrics
            metrics = f"""
╔══════════════════════════════════════════════════════════╗
║        MODEL: {model_type.upper():^40}        ║
╚══════════════════════════════════════════════════════════╝

📊 TRAINING SET PERFORMANCE
  ├─ R² Score:              {train_r2:.4f}
  ├─ Mean Absolute Error:   ${train_mae:,.2f}
  └─ Root Mean Squared Error: ${train_rmse:,.2f}

📈 TESTING SET PERFORMANCE
  ├─ R² Score:              {test_r2:.4f}
  ├─ Mean Absolute Error:   ${test_mae:,.2f}
  └─ Root Mean Squared Error: ${test_rmse:,.2f}

🔄 CROSS-VALIDATION (5-Fold)
  ├─ Mean CV R² Score:      {cv_scores.mean():.4f}
  ├─ Std Deviation:         {cv_scores.std():.4f}
  └─ CV Scores:             {', '.join([f'{s:.3f}' for s in cv_scores])}

✅ Model Performance: {"Excellent" if test_r2 > 0.9 else "Good" if test_r2 > 0.7 else "Fair"}
📍 Overfitting Check:  {"✅ Good" if abs(train_r2 - test_r2) < 0.1 else "⚠️ Possible Overfitting"}
            """

            self.metrics_text.delete('1.0', tk.END)
            self.metrics_text.insert('1.0', metrics)

            self.status_bar.config(text=f"✅ {model_type} trained | Test R²: {test_r2:.4f}")

            messagebox.showinfo("Success",
                                f"✅ Model trained successfully!\n\nTest R² Score: {test_r2:.4f}\nTest MAE: ${test_mae:,.2f}")

        except Exception as e:
            messagebox.showerror("Error", f"Training failed:\n{e}")

    def plot_regression(self):
        """Plot regression line"""
        if self.current_model is None:
            messagebox.showwarning("Warning", "Please train a model first!")
            return

        # Clear previous plot
        for widget in self.plot_frame.winfo_children():
            widget.destroy()

        fig = Figure(figsize=(10, 6))
        ax = fig.add_subplot(111)

        # Plot training data
        ax.scatter(self.X_train, self.y_train, color='#3498db',
                   label='Training Data', alpha=0.6, s=100, edgecolors='black')
        ax.scatter(self.X_test, self.y_test, color='#e74c3c',
                   label='Test Data', alpha=0.6, s=100, edgecolors='black')

        # Plot regression line
        X_range = np.linspace(self.df['YearsExperience'].min(),
                              self.df['YearsExperience'].max(), 100).reshape(-1, 1)

        if 'Polynomial' in self.model_name.get():
            degree = 2 if "Degree 2" in self.model_name.get() else 3
            poly = self.models[f'poly_{degree}']
            X_range_transformed = poly.transform(X_range)
        else:
            X_range_transformed = X_range

        y_range = self.current_model.predict(X_range_transformed)
        ax.plot(X_range, y_range, color='#2ecc71', linewidth=3,
                label='Regression Line')

        ax.set_xlabel('Years of Experience', fontsize=12, fontweight='bold')
        ax.set_ylabel('Salary ($)', fontsize=12, fontweight='bold')
        ax.set_title(f'Salary Prediction - {self.model_name.get()}',
                     fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

        canvas = FigureCanvasTkAgg(fig, master=self.plot_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill='both', expand=True)

    def plot_residuals(self):
        """Plot residuals"""
        if self.current_model is None:
            messagebox.showwarning("Warning", "Please train a model first!")
            return

        for widget in self.plot_frame.winfo_children():
            widget.destroy()

        # Get predictions
        if 'Polynomial' in self.model_name.get():
            degree = 2 if "Degree 2" in self.model_name.get() else 3
            poly = self.models[f'poly_{degree}']
            X_test_transformed = poly.transform(self.X_test)
        else:
            X_test_transformed = self.X_test

        y_pred = self.current_model.predict(X_test_transformed)
        residuals = self.y_test - y_pred

        fig = Figure(figsize=(10, 6))
        ax = fig.add_subplot(111)

        ax.scatter(y_pred, residuals, color='#9b59b6', alpha=0.6,
                   s=100, edgecolors='black')
        ax.axhline(y=0, color='red', linestyle='--', linewidth=2)
        ax.set_xlabel('Predicted Values', fontsize=12, fontweight='bold')
        ax.set_ylabel('Residuals', fontsize=12, fontweight='bold')
        ax.set_title('Residual Plot', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)

        canvas = FigureCanvasTkAgg(fig, master=self.plot_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill='both', expand=True)

    def plot_distribution(self):
        """Plot salary distribution"""
        if self.df is None:
            messagebox.showwarning("Warning", "Please load data first!")
            return

        for widget in self.plot_frame.winfo_children():
            widget.destroy()

        fig = Figure(figsize=(10, 6))
        ax = fig.add_subplot(111)

        ax.hist(self.df['Salary'], bins=15, color='#e74c3c',
                alpha=0.7, edgecolor='black')
        ax.set_xlabel('Salary ($)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Frequency', fontsize=12, fontweight='bold')
        ax.set_title('Salary Distribution', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')

        canvas = FigureCanvasTkAgg(fig, master=self.plot_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill='both', expand=True)

    def plot_actual_vs_predicted(self):
        """Plot actual vs predicted values"""
        if self.current_model is None:
            messagebox.showwarning("Warning", "Please train a model first!")
            return

        for widget in self.plot_frame.winfo_children():
            widget.destroy()

        if 'Polynomial' in self.model_name.get():
            degree = 2 if "Degree 2" in self.model_name.get() else 3
            poly = self.models[f'poly_{degree}']
            X_test_transformed = poly.transform(self.X_test)
        else:
            X_test_transformed = self.X_test

        y_pred = self.current_model.predict(X_test_transformed)

        fig = Figure(figsize=(10, 6))
        ax = fig.add_subplot(111)

        ax.scatter(self.y_test, y_pred, color='#1abc9c', alpha=0.6,
                   s=100, edgecolors='black')

        # Perfect prediction line
        min_val = min(self.y_test.min(), y_pred.min())
        max_val = max(self.y_test.max(), y_pred.max())
        ax.plot([min_val, max_val], [min_val, max_val],
                'r--', linewidth=2, label='Perfect Prediction')

        ax.set_xlabel('Actual Salary ($)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Predicted Salary ($)', fontsize=12, fontweight='bold')
        ax.set_title('Actual vs Predicted Salaries', fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

        canvas = FigureCanvasTkAgg(fig, master=self.plot_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill='both', expand=True)

    def plot_all(self):
        """Display all plots in a grid"""
        if self.current_model is None:
            messagebox.showwarning("Warning", "Please train a model first!")
            return

        for widget in self.plot_frame.winfo_children():
            widget.destroy()

        fig = Figure(figsize=(12, 8))

        # Get predictions
        if 'Polynomial' in self.model_name.get():
            degree = 2 if "Degree 2" in self.model_name.get() else 3
            poly = self.models[f'poly_{degree}']
            X_test_transformed = poly.transform(self.X_test)
            X_range = np.linspace(self.df['YearsExperience'].min(),
                                  self.df['YearsExperience'].max(), 100).reshape(-1, 1)
            X_range_transformed = poly.transform(X_range)
        else:
            X_test_transformed = self.X_test
            X_range = np.linspace(self.df['YearsExperience'].min(),
                                  self.df['YearsExperience'].max(), 100).reshape(-1, 1)
            X_range_transformed = X_range

        y_pred = self.current_model.predict(X_test_transformed)
        y_range = self.current_model.predict(X_range_transformed)
        residuals = self.y_test - y_pred

        # Plot 1: Regression
        ax1 = fig.add_subplot(2, 2, 1)
        ax1.scatter(self.X_train, self.y_train, color='#3498db', alpha=0.6, label='Train')
        ax1.scatter(self.X_test, self.y_test, color='#e74c3c', alpha=0.6, label='Test')
        ax1.plot(X_range, y_range, color='#2ecc71', linewidth=2, label='Prediction')
        ax1.set_xlabel('Experience (years)')
        ax1.set_ylabel('Salary ($)')
        ax1.set_title('Regression Line')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Plot 2: Residuals
        ax2 = fig.add_subplot(2, 2, 2)
        ax2.scatter(y_pred, residuals, color='#9b59b6', alpha=0.6)
        ax2.axhline(y=0, color='red', linestyle='--', linewidth=2)
        ax2.set_xlabel('Predicted Values')
        ax2.set_ylabel('Residuals')
        ax2.set_title('Residual Plot')
        ax2.grid(True, alpha=0.3)

        # Plot 3: Distribution
        ax3 = fig.add_subplot(2, 2, 3)
        ax3.hist(self.df['Salary'], bins=15, color='#e74c3c', alpha=0.7, edgecolor='black')
        ax3.set_xlabel('Salary ($)')
        ax3.set_ylabel('Frequency')
        ax3.set_title('Salary Distribution')
        ax3.grid(True, alpha=0.3, axis='y')

        # Plot 4: Actual vs Predicted
        ax4 = fig.add_subplot(2, 2, 4)
        ax4.scatter(self.y_test, y_pred, color='#1abc9c', alpha=0.6)
        min_val = min(self.y_test.min(), y_pred.min())
        max_val = max(self.y_test.max(), y_pred.max())
        ax4.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect')
        ax4.set_xlabel('Actual Salary ($)')
        ax4.set_ylabel('Predicted Salary ($)')
        ax4.set_title('Actual vs Predicted')
        ax4.legend()
        ax4.grid(True, alpha=0.3)

        fig.tight_layout()

        canvas = FigureCanvasTkAgg(fig, master=self.plot_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill='both', expand=True)

    def predict_salary(self):
        """Predict salary for given experience"""
        if self.current_model is None:
            messagebox.showwarning("Warning", "Please train a model first!")
            return

        try:
            exp = float(self.entry_exp.get())

            if exp < 0:
                messagebox.showerror("Error", "Experience cannot be negative!")
                return

            X_pred = np.array([[exp]])

            if 'Polynomial' in self.model_name.get():
                degree = 2 if "Degree 2" in self.model_name.get() else 3
                poly = self.models[f'poly_{degree}']
                X_pred = poly.transform(X_pred)

            pred = self.current_model.predict(X_pred)[0]

            # Display result
            result_text = f"💰 Predicted Salary: ${pred:,.2f}\n"
            result_text += f"📊 Model: {self.model_name.get()}\n"
            result_text += f"📈 Experience: {exp:.1f} years"

            self.lbl_result.config(text=result_text, fg='#27ae60')

            # Add to history
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            self.history.append((exp, pred, self.model_name.get(), timestamp))
            self.history_tree.insert('', 0, values=(
                f"{exp:.1f} years",
                f"${pred:,.2f}",
                self.model_name.get(),
                timestamp
            ))

            self.status_bar.config(text=f"✅ Prediction complete | ${pred:,.2f}")

        except ValueError:
            messagebox.showerror("Error", "Please enter a valid number!")

    def compare_models(self):
        """Compare all models"""
        if self.df is None:
            messagebox.showwarning("Warning", "Please load data first!")
            return

        try:
            X = self.df[['YearsExperience']].values
            y = self.df['Salary'].values

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42)

            models_to_compare = {
                'Linear Regression': (LinearRegression(), X_train, X_test),
                'Polynomial (Deg 2)': (LinearRegression(),
                                       PolynomialFeatures(2).fit_transform(X_train),
                                       PolynomialFeatures(2).fit_transform(X_test)),
                'Polynomial (Deg 3)': (LinearRegression(),
                                       PolynomialFeatures(3).fit_transform(X_train),
                                       PolynomialFeatures(3).fit_transform(X_test)),
                'Ridge': (Ridge(alpha=1.0), X_train, X_test),
                'Lasso': (Lasso(alpha=1.0), X_train, X_test)
            }

            results = []

            for name, (model, X_tr, X_te) in models_to_compare.items():
                model.fit(X_tr, y_train)
                y_pred = model.predict(X_te)

                r2 = r2_score(y_test, y_pred)
                mae = mean_absolute_error(y_test, y_pred)
                rmse = np.sqrt(mean_squared_error(y_test, y_pred))

                results.append({
                    'Model': name,
                    'R² Score': r2,
                    'MAE': mae,
                    'RMSE': rmse
                })

            # Clear comparison frame
            for widget in self.comparison_frame.winfo_children():
                widget.destroy()

            # Create comparison plot
            fig = Figure(figsize=(12, 8))

            # R² Score comparison
            ax1 = fig.add_subplot(2, 2, 1)
            models_names = [r['Model'] for r in results]
            r2_scores = [r['R² Score'] for r in results]
            colors = ['#3498db', '#9b59b6', '#e74c3c', '#2ecc71', '#f39c12']
            bars1 = ax1.bar(models_names, r2_scores, color=colors, edgecolor='black', linewidth=1.5)
            ax1.set_ylabel('R² Score', fontweight='bold')
            ax1.set_title('R² Score Comparison', fontweight='bold', fontsize=12)
            ax1.set_ylim([0, 1])
            ax1.tick_params(axis='x', rotation=45)
            ax1.grid(True, alpha=0.3, axis='y')

            # Add value labels on bars
            for bar in bars1:
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width() / 2., height,
                         f'{height:.3f}', ha='center', va='bottom', fontweight='bold')

            # MAE comparison
            ax2 = fig.add_subplot(2, 2, 2)
            mae_scores = [r['MAE'] for r in results]
            bars2 = ax2.bar(models_names, mae_scores, color=colors, edgecolor='black', linewidth=1.5)
            ax2.set_ylabel('Mean Absolute Error ($)', fontweight='bold')
            ax2.set_title('MAE Comparison (Lower is Better)', fontweight='bold', fontsize=12)
            ax2.tick_params(axis='x', rotation=45)
            ax2.grid(True, alpha=0.3, axis='y')

            for bar in bars2:
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width() / 2., height,
                         f'${height:,.0f}', ha='center', va='bottom', fontweight='bold', fontsize=8)

            # RMSE comparison
            ax3 = fig.add_subplot(2, 2, 3)
            rmse_scores = [r['RMSE'] for r in results]
            bars3 = ax3.bar(models_names, rmse_scores, color=colors, edgecolor='black', linewidth=1.5)
            ax3.set_ylabel('Root Mean Squared Error ($)', fontweight='bold')
            ax3.set_title('RMSE Comparison (Lower is Better)', fontweight='bold', fontsize=12)
            ax3.tick_params(axis='x', rotation=45)
            ax3.grid(True, alpha=0.3, axis='y')

            for bar in bars3:
                height = bar.get_height()
                ax3.text(bar.get_x() + bar.get_width() / 2., height,
                         f'${height:,.0f}', ha='center', va='bottom', fontweight='bold', fontsize=8)

            # Summary table
            ax4 = fig.add_subplot(2, 2, 4)
            ax4.axis('off')

            table_data = []
            for r in results:
                table_data.append([
                    r['Model'],
                    f"{r['R² Score']:.4f}",
                    f"${r['MAE']:,.0f}",
                    f"${r['RMSE']:,.0f}"
                ])

            table = ax4.table(cellText=table_data,
                              colLabels=['Model', 'R²', 'MAE', 'RMSE'],
                              cellLoc='center',
                              loc='center',
                              colWidths=[0.35, 0.2, 0.225, 0.225])

            table.auto_set_font_size(False)
            table.set_fontsize(9)
            table.scale(1, 2)

            # Style header
            for i in range(4):
                cell = table[(0, i)]
                cell.set_facecolor('#34495e')
                cell.set_text_props(weight='bold', color='white')

            # Highlight best model (highest R²)
            best_idx = r2_scores.index(max(r2_scores))
            for i in range(4):
                cell = table[(best_idx + 1, i)]
                cell.set_facecolor('#2ecc71')
                cell.set_text_props(weight='bold')

            ax4.set_title('📊 Model Performance Summary', fontweight='bold', fontsize=12, pad=20)

            fig.tight_layout()

            canvas = FigureCanvasTkAgg(fig, master=self.comparison_frame)
            canvas.draw()
            canvas.get_tk_widget().pack(fill='both', expand=True)

            # Find best model
            best_model = results[best_idx]['Model']
            best_r2 = results[best_idx]['R² Score']

            messagebox.showinfo("Comparison Complete",
                                f"✅ Model comparison complete!\n\n🏆 Best Model: {best_model}\n📊 R² Score: {best_r2:.4f}")

            self.status_bar.config(text=f"✅ Comparison complete | Best: {best_model} (R²: {best_r2:.4f})")

        except Exception as e:
            messagebox.showerror("Error", f"Comparison failed:\n{e}")

    def export_report(self):
        """Export analysis report"""
        if self.current_model is None:
            messagebox.showwarning("Warning", "Please train a model first!")
            return

        try:
            file_path = filedialog.asksaveasfilename(
                defaultextension=".txt",
                filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
            )

            if not file_path:
                return

            # Get predictions
            if 'Polynomial' in self.model_name.get():
                degree = 2 if "Degree 2" in self.model_name.get() else 3
                poly = self.models[f'poly_{degree}']
                X_test_transformed = poly.transform(self.X_test)
            else:
                X_test_transformed = self.X_test

            y_pred = self.current_model.predict(X_test_transformed)

            # Calculate metrics
            r2 = r2_score(self.y_test, y_pred)
            mae = mean_absolute_error(self.y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(self.y_test, y_pred))

            with open(file_path, 'w') as f:
                f.write("=" * 60 + "\n")
                f.write(" SALARY PREDICTION MODEL - ANALYSIS REPORT\n")
                f.write("=" * 60 + "\n\n")
                f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

                f.write("MODEL INFORMATION\n")
                f.write("-" * 60 + "\n")
                f.write(f"Model Type: {self.model_name.get()}\n")
                f.write(f"Training Samples: {len(self.X_train)}\n")
                f.write(f"Testing Samples: {len(self.X_test)}\n\n")

                f.write("PERFORMANCE METRICS\n")
                f.write("-" * 60 + "\n")
                f.write(f"R² Score:                 {r2:.4f}\n")
                f.write(f"Mean Absolute Error:      ${mae:,.2f}\n")
                f.write(f"Root Mean Squared Error:  ${rmse:,.2f}\n\n")

                f.write("DATASET STATISTICS\n")
                f.write("-" * 60 + "\n")
                f.write(
                    f"Experience Range: {self.df['YearsExperience'].min():.1f} - {self.df['YearsExperience'].max():.1f} years\n")
                f.write(f"Salary Range:     ${self.df['Salary'].min():,.2f} - ${self.df['Salary'].max():,.2f}\n")
                f.write(f"Mean Salary:      ${self.df['Salary'].mean():,.2f}\n")
                f.write(f"Median Salary:    ${self.df['Salary'].median():,.2f}\n\n")

                if self.history:
                    f.write("PREDICTION HISTORY\n")
                    f.write("-" * 60 + "\n")
                    f.write(f"{'Experience':<15} {'Predicted Salary':<20} {'Timestamp'}\n")
                    f.write("-" * 60 + "\n")
                    for exp, salary, _, timestamp in self.history[-20:]:
                        f.write(f"{exp:>10.1f} years  ${salary:>15,.2f}  {timestamp}\n")

                f.write("\n" + "=" * 60 + "\n")
                f.write("End of Report\n")
                f.write("=" * 60 + "\n")

            messagebox.showinfo("Success", f"Report exported successfully to:\n{file_path}")
            self.status_bar.config(text=f"✅ Report exported to {file_path}")

        except Exception as e:
            messagebox.showerror("Error", f"Export failed:\n{e}")


# ============ Main Application ============
if __name__ == "__main__":
    root = tk.Tk()
    app = SalaryPredictionApp(root)
    root.mainloop()