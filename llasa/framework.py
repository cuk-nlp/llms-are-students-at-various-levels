import os
import sys
import pandas as pd
import glob
import ast
import numpy as np
import heapq
import random
from sklearn.metrics import mean_squared_error, mean_absolute_error, f1_score, accuracy_score, precision_score, recall_score
import scipy.stats as stats
import warnings
warnings.filterwarnings('ignore')


class Reporter:
    def sort_dict_by_key(self, d):
        return dict(sorted(d.items()))

    def classify_into_bins(self, values, bins):
        bin_indices = np.digitize(values, bins)
        return bin_indices

    def classification_setting_compute_metrics(self, y_true, y_pred):
        f1 = f1_score(y_true, y_pred, average='weighted')
        acc = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average='weighted')
        recall = recall_score(y_true, y_pred, average='weighted')
        return f1, acc, precision, recall

    def log_results(self, result_df, test_dataset, llasa_type, is_llmda, classification_setting_num):
        # Check for incompatible parameters
        if llasa_type == "zero" and is_llmda:
            raise ValueError("Zero-shot LLaSA cannot be used with LLMDA.")

        # Extract classification labels from the test dataset
        classification_labels = {q: v for q, v in zip(test_dataset['question_id'], test_dataset['pred_irt'])}
        classification_label_edges = np.histogram_bin_edges(list(classification_labels.values()), bins=classification_setting_num-1)
        classification_labels = self.classify_into_bins(list(classification_labels.values()), classification_label_edges)

        # Get the predicted difficulty from the result dataframe
        result_df = result_df.sort_values(by=['train_rmse'])
        result = result_df.iloc[0]
        preds = self.sort_dict_by_key(ast.literal_eval(result['test_difficulty']))
        preds = self.classify_into_bins(list(preds.values()), classification_label_edges)

        f1, acc, precision, recall = self.classification_setting_compute_metrics(classification_labels, preds)
        print(f"\n{'#' * 20} Settings {'#' * 20}\n")
        print(f"LLaSA Type: {llasa_type}")
        print(f"LLMDA: {is_llmda}")
        print(f"Classification Settings: {classification_setting_num} classes")
        
        # Train Metrics
        print(f"\n{'#' * 20} Train Metrics {'#' * 20}\n")
        print(f"Train RMSE: {result['train_rmse']:.3f}")
        print(f"Train MAE: {result['train_mae']:.3f}")
        
        # Test Metrics
        print(f"\n{'#' * 20} Test Metrics {'#' * 20}\n")
        print(f"Test RMSE: {result['test_rmse']:.3f}")
        print(f"Test MAE: {result['test_mae']:.3f}")
        print(f"\nTest F1-score: {f1:.3f}")
        print(f"Test Accuracy: {acc:.3f}")
        print(f"Test Precision: {precision:.3f}")
        print(f"Test Recall: {recall:.3f}")
        print(f"\nTest Correlation: {result['test_corr']:.3f}")
        print(f"Test Correlation p-value: {result['test_corr_p']:.3f}")
        print(f"\n{'#' * 54}")

    
class IRTFramework:
    def calculate_metrics(self, y_true, y_pred):
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_true, y_pred)
        corr_result = stats.pearsonr(y_true, y_pred)
        return mse, rmse, mae, corr_result.statistic, corr_result.pvalue

    def delete_trash_files(self, save_dir):
        trash_files = glob.glob(os.path.join(save_dir, '*trash*'))
        for file_path in trash_files:
            try:
                os.remove(file_path)
            except Exception as e:
                print(f"Failed to delete {file_path}: {e}")

    def get_irt(self, df, save_dir, data_type, train_type, is_remove=False):
        """
        Executes IRT analysis using an R script and retrieves the ability and difficulty results.

        Args:
            df (pd.DataFrame): Input data for IRT analysis.
            save_dir (str): Directory to save the results.
            data_type (str): Data type identifier (e.g., 'dk').
            train_type (str): Type of dataset (e.g., 'train' or 'test').
            is_remove (bool): Whether to remove temporary files after execution.

        Returns:
            tuple: Dictionaries for ability and difficulty from the IRT results.
        """
        # Save the DataFrame to a CSV file
        df.to_csv(f'{save_dir}/{data_type}_{train_type}.csv', index=False)

        # Run the R script for IRT
        os.system(f"Rscript ./llasa/get_irt.r -o {save_dir} -d {data_type} -t {train_type} > /dev/null 2>&1")

        # Load ability and difficulty from the generated CSV files
        ability = pd.read_csv(f'{save_dir}/{data_type}_{train_type}_ability.csv')
        ability = {int(a['Unnamed: 0']): a['F1'] for _, a in ability.iterrows()}

        difficulty = pd.read_csv(f'{save_dir}/{data_type}_{train_type}_difficulty.csv')
        difficulty = {int(d['Unnamed: 0']): d['x'] for _, d in difficulty.iterrows()}

        ability = {df.columns[i-1]:a for i, a in ability.items()}
        # Optionally delete temporary 'trash' files
        if is_remove:
            self.delete_trash_files(save_dir)

        return ability, difficulty


class LLaSA(IRTFramework):
    def student_representative_llm_cluster_selection(self, student_response_df, student_ability_dict, model_ability_dict, cluster_size):
        """
        Selects the closest models for each student based on their ability.

        Args:
            student_response_df (pd.DataFrame): DataFrame containing student responses.
            student_ability_dict (dict): Student abilities.
            model_ability_dict (dict): Model abilities.
            cluster_size (int): Number of models to cluster.

        Returns:
            dict: Dictionary mapping students to their closest models.
        """
        student_model_pair_list = {}
        for student_name in student_response_df.drop("question_id", axis=1).columns:
            closest_models = heapq.nsmallest(
                cluster_size,
                model_ability_dict.items(),
                key=lambda x: self.distance(x[1], student_ability_dict[student_name])
            )
            closest_models_list = [model[0] for model in closest_models]
            student_model_pair_list[student_name] = closest_models_list

        return student_model_pair_list

    def llm_cluster_response_aggregation(self, student_model_pair_list, model_response_df):
        """
        Aggregates responses from closest models for each student.

        Args:
            student_model_pair_list (dict): Dictionary mapping students to their closest models.
            model_response_df (pd.DataFrame): Model response data.

        Returns:
            pd.DataFrame: Aggregated responses for each student.
        """
        aggregation_result_list = []
        for student_name, closest_models_list in student_model_pair_list.items():
            closest_models_response = np.sum(model_response_df[closest_models_list].values, axis=-1)
            closest_models_response = np.clip(closest_models_response, 0, 1)
            aggregation_result_list.append(closest_models_response.astype(int))

        return pd.DataFrame(np.array(aggregation_result_list).T, index=model_response_df.index, columns=student_model_pair_list.keys())

    def distance(self, x1, x2):
        return abs(x1 - x2)


class ZeroshotLLaSA(IRTFramework):
    def __init__(self, accuracy_df, level_llms, low_ratio=0.25, high_ratio=0.75):
        """
        Initializes ZeroLLaSA by setting up accuracy thresholds and splitting models into categories.

        Args:
            accuracy_df (pd.DataFrame): DataFrame with accuracy values for different models.
            level_llms (dict): Dictionary specifying the number of models to select for each ability level.
            low_ratio (float): Threshold for 'low' ability models.
            high_ratio (float): Threshold for 'high' ability models.
        """
        super().__init__()
        self.accuracy_df = accuracy_df
        self.level_llms = level_llms
        self.low_ratio = low_ratio
        self.high_ratio = high_ratio

        self.setting()

    def setting(self):
        self.low_threshold = self.accuracy_df['value'].quantile(self.low_ratio)
        self.high_threshold = self.accuracy_df['value'].quantile(self.high_ratio)
        self.accuracy_df['group'] = self.accuracy_df['value'].apply(self.categorize)

        self.low_dict = self.accuracy_df[self.accuracy_df['group'] == 'low'].set_index('key')['value'].to_dict()
        self.middle_dict = self.accuracy_df[self.accuracy_df['group'] == 'middle'].set_index('key')['value'].to_dict()
        self.high_dict = self.accuracy_df[self.accuracy_df['group'] == 'high'].set_index('key')['value'].to_dict()

    def categorize(self, value):
        if value <= self.low_threshold:
            return 'low'
        elif value <= self.high_threshold:
            return 'middle'
        return 'high'

    def llm_selection(self):
        """Selects models from low, middle, and high groups based on the provided ratio."""
        rep_model_list = []
        
        low_keys = random.sample(list(self.low_dict.keys()), min(self.level_llms['low'], len(self.low_dict)))
        middle_keys = random.sample(list(self.middle_dict.keys()), min(self.level_llms['middle'], len(self.middle_dict)))
        high_keys = random.sample(list(self.high_dict.keys()), min(self.level_llms['high'], len(self.high_dict)))

        rep_model_list.extend(low_keys)
        rep_model_list.extend(middle_keys)
        rep_model_list.extend(high_keys)

        return rep_model_list
