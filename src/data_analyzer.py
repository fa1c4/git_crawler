"""
GitHub Repository Data Analyzer
Analyzes GitHub repository data collected by github_crawler.py
"""

import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import logging
from typing import Dict, List, Optional, Tuple
from tabulate import tabulate
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class GitHubDataAnalyzer:
    def __init__(self, data_path: str = '../results'):
        """
        Initialize the analyzer with data path
        
        Args:
            data_path: Path to directory containing JSON data files
        """
        self.data_path = Path(data_path)
        self.df = None
        self.metadata = None
        
        # Set up plotting style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
    def load_data(self, filename: str = None) -> bool:
        """
        Load GitHub repository data from JSON file
        
        Args:
            filename: Specific filename to load. If None, tries to find available files
            
        Returns:
            bool: True if data loaded successfully, False otherwise
        """
        try:
            if filename:
                file_path = self.data_path / filename
            else:
                # Try to find available data files in order of preference
                potential_files = [
                    'github_repos.json',
                    'github_repos_progress.json', 
                    'github_repos_summary.json',
                    # 'multilang_github_repos.json',
                    # 'multilang_github_repos_progress.json', 
                    # 'multilang_github_repos_summary.json',
                    # 'py_github_repos.json',
                    # 'py_github_repos_progress.json', 
                    # 'py_github_repos_summary.json',

                ]
                
                file_path = None
                for file in potential_files:
                    if (self.data_path / file).exists():
                        file_path = self.data_path / file
                        logger.info(f"Found data file: {file}")
                        break
                        
                if not file_path:
                    logger.error(f"No data files found in {self.data_path}")
                    return False
            
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Handle different data formats
            if 'repositories' in data and isinstance(data['repositories'], list):
                # Standard format from github_repos.json or progress file
                repos_data = data['repositories']
                self.metadata = data.get('metadata', {})
            elif isinstance(data, list):
                # Direct list of repositories
                repos_data = data
                self.metadata = {}
            else:
                logger.error("Unrecognized data format")
                return False
            
            # Convert to DataFrame
            self.df = pd.DataFrame(repos_data)
            
            # Clean and prepare data
            self._clean_data()
            
            logger.info(f"Loaded {len(self.df)} repositories from {file_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            return False
    
    def _clean_data(self):
        """Clean and prepare the data for analysis"""
        if self.df is None:
            return
            
        # Handle missing values
        numeric_columns = ['stars', 'forks', 'watchers', 'issues_open', 'issues_closed', 
                          'pr_open', 'pr_closed', 'contributors', 'commits', 'releases', 'readme_lines']
        
        for col in numeric_columns:
            if col in self.df.columns:
                self.df[col] = pd.to_numeric(self.df[col], errors='coerce').fillna(0)
        
        # Clean language data
        if 'language' in self.df.columns:
            self.df['language'] = self.df['language'].fillna('Unknown')
            self.df['language'] = self.df['language'].replace('', 'Unknown')
        
        # Clean license data
        if 'license' in self.df.columns:
            self.df['license'] = self.df['license'].fillna('None')
            self.df['license'] = self.df['license'].replace('', 'None')
        
        # Add calculated fields
        if 'issues_open' in self.df.columns and 'issues_closed' in self.df.columns:
            self.df['total_issues'] = self.df['issues_open'] + self.df['issues_closed']
        
        if 'pr_open' in self.df.columns and 'pr_closed' in self.df.columns:
            self.df['total_prs'] = self.df['pr_open'] + self.df['pr_closed']
        
        # Calculate ratios
        self.df['fork_to_star_ratio'] = np.where(self.df['stars'] > 0, 
                                               self.df['forks'] / self.df['stars'], 0)
        
        # Create popularity score (weighted combination of metrics)
        self.df['popularity_score'] = (
            self.df['stars'] * 0.4 + 
            self.df['forks'] * 0.3 + 
            self.df['contributors'] * 0.2 + 
            self.df['watchers'] * 0.1
        )
    
    def generate_basic_statistics(self) -> Dict:
        """Generate basic statistics about the dataset"""
        if self.df is None:
            return {}
        
        stats = {
            'total_repositories': len(self.df),
            'total_stars': int(self.df['stars'].sum()),
            'total_forks': int(self.df['forks'].sum()),
            'total_contributors': int(self.df['contributors'].sum()),
            'average_stars': round(self.df['stars'].mean(), 2),
            'median_stars': int(self.df['stars'].median()),
            'average_forks': round(self.df['forks'].mean(), 2),
            'average_contributors': round(self.df['contributors'].mean(), 2),
            'unique_languages': len(self.df['language'].unique()),
            'unique_licenses': len(self.df['license'].unique()),
        }
        
        if 'ci_cd' in self.df.columns:
            stats['repositories_with_ci_cd'] = int(self.df['ci_cd'].sum())
            stats['ci_cd_percentage'] = round((self.df['ci_cd'].sum() / len(self.df)) * 100, 2)
        
        return stats
    
    def analyze_stars_distribution(self) -> Dict:
        """Analyze the distribution of stars across repositories"""
        if self.df is None or 'stars' not in self.df.columns:
            return {}
        
        stars = self.df['stars']
        
        # Define star ranges
        ranges = [
            (0, 100), (100, 500), (500, 1000), (1000, 5000), 
            (5000, 10000), (10000, 50000), (50000, 100000), (100000, float('inf'))
        ]
        
        distribution = {}
        for min_stars, max_stars in ranges:
            if max_stars == float('inf'):
                count = len(stars[stars >= min_stars])
                label = f"{min_stars}+"
            else:
                count = len(stars[(stars >= min_stars) & (stars < max_stars)])
                label = f"{min_stars}-{max_stars}"
            
            distribution[label] = count
        
        return {
            'distribution': distribution,
            'percentiles': {
                '25th': int(stars.quantile(0.25)),
                '50th': int(stars.quantile(0.50)),
                '75th': int(stars.quantile(0.75)),
                '90th': int(stars.quantile(0.90)),
                '95th': int(stars.quantile(0.95)),
                '99th': int(stars.quantile(0.99))
            },
            'top_starred': self.df.nlargest(10, 'stars')[['repo_name', 'owner', 'language', 'stars']].to_dict('records')
        }
    
    def analyze_language_distribution(self) -> Dict:
        """Analyze programming language distribution and metrics"""
        if self.df is None or 'language' not in self.df.columns:
            return {}
        
        language_stats = self.df.groupby('language').agg({
            'repo_name': 'count',
            'stars': ['sum', 'mean', 'median'],
            'forks': ['sum', 'mean'],
            'contributors': ['sum', 'mean'],
            'total_issues': 'sum' if 'total_issues' in self.df.columns else 'mean',
            'ci_cd': 'sum' if 'ci_cd' in self.df.columns else 'mean'
        }).round(2)
        
        # Flatten column names
        language_stats.columns = ['_'.join(col).strip() for col in language_stats.columns.values]
        language_stats = language_stats.reset_index()
        
        # Rename columns for clarity
        rename_map = {
            'repo_name_count': 'repositories',
            'stars_sum': 'total_stars',
            'stars_mean': 'avg_stars',
            'stars_median': 'median_stars',
            'forks_sum': 'total_forks',
            'forks_mean': 'avg_forks',
            'contributors_sum': 'total_contributors',
            'contributors_mean': 'avg_contributors'
        }
        
        language_stats = language_stats.rename(columns=rename_map)
        language_stats = language_stats.sort_values('repositories', ascending=False)
        
        return {
            'language_stats': language_stats.to_dict('records'),
            'top_languages_by_repos': language_stats.head(10),
            'top_languages_by_stars': language_stats.nlargest(10, 'total_stars')
        }
    
    def analyze_license_distribution(self) -> Dict:
        """Analyze license distribution"""
        if self.df is None or 'license' not in self.df.columns:
            return {}
        
        license_counts = self.df['license'].value_counts()
        license_percentages = (license_counts / len(self.df) * 100).round(2)
        
        return {
            'license_distribution': dict(license_counts),
            'license_percentages': dict(license_percentages),
            'top_licenses': license_counts.head(10).to_dict()
        }
    
    def analyze_activity_metrics(self) -> Dict:
        """Analyze repository activity metrics"""
        if self.df is None:
            return {}
        
        metrics = {}
        
        # Issues analysis
        if 'total_issues' in self.df.columns:
            metrics['issues'] = {
                'avg_total_issues': round(self.df['total_issues'].mean(), 2),
                'median_total_issues': int(self.df['total_issues'].median()),
                'max_issues': int(self.df['total_issues'].max()),
                'repos_with_issues': int((self.df['total_issues'] > 0).sum()),
                'percentage_with_issues': round((self.df['total_issues'] > 0).mean() * 100, 2)
            }
        
        # Pull requests analysis
        if 'total_prs' in self.df.columns:
            metrics['pull_requests'] = {
                'avg_total_prs': round(self.df['total_prs'].mean(), 2),
                'median_total_prs': int(self.df['total_prs'].median()),
                'max_prs': int(self.df['total_prs'].max()),
                'repos_with_prs': int((self.df['total_prs'] > 0).sum()),
                'percentage_with_prs': round((self.df['total_prs'] > 0).mean() * 100, 2)
            }
        
        # Contributors analysis
        if 'contributors' in self.df.columns:
            metrics['contributors'] = {
                'avg_contributors': round(self.df['contributors'].mean(), 2),
                'median_contributors': int(self.df['contributors'].median()),
                'max_contributors': int(self.df['contributors'].max()),
                'single_contributor_repos': int((self.df['contributors'] == 1).sum()),
                'large_contributor_repos': int((self.df['contributors'] >= 100).sum())
            }
        
        return metrics
    
    def print_summary_tables(self):
        """Print formatted summary tables"""
        if self.df is None:
            print("No data loaded. Please load data first.")
            return
        
        print("=" * 80)
        print("GITHUB REPOSITORY DATA ANALYSIS SUMMARY")
        print("=" * 80)
        
        # Basic statistics
        basic_stats = self.generate_basic_statistics()
        print("\n📊 BASIC STATISTICS")
        print("-" * 40)
        
        stats_table = [
            ["Total Repositories", f"{basic_stats['total_repositories']:,}"],
            ["Total Stars", f"{basic_stats['total_stars']:,}"],
            ["Total Forks", f"{basic_stats['total_forks']:,}"],
            ["Total Contributors", f"{basic_stats['total_contributors']:,}"],
            ["Average Stars", f"{basic_stats['average_stars']:,}"],
            ["Median Stars", f"{basic_stats['median_stars']:,}"],
            ["Average Forks", f"{basic_stats['average_forks']:,}"],
            ["Average Contributors", f"{basic_stats['average_contributors']:,}"],
            ["Unique Languages", f"{basic_stats['unique_languages']}"],
            ["Unique Licenses", f"{basic_stats['unique_licenses']}"]
        ]
        
        if 'ci_cd_percentage' in basic_stats:
            stats_table.extend([
                ["Repos with CI/CD", f"{basic_stats['repositories_with_ci_cd']:,}"],
                ["CI/CD Percentage", f"{basic_stats['ci_cd_percentage']}%"]
            ])
        
        print(tabulate(stats_table, headers=["Metric", "Value"], tablefmt="grid"))
        
        # Stars distribution
        stars_analysis = self.analyze_stars_distribution()
        if stars_analysis:
            print("\n⭐ STARS DISTRIBUTION")
            print("-" * 40)
            
            dist_table = [[range_name, count] for range_name, count in stars_analysis['distribution'].items()]
            print(tabulate(dist_table, headers=["Star Range", "Repositories"], tablefmt="grid"))
            
            print("\n📈 STARS PERCENTILES")
            print("-" * 40)
            percentiles_table = [[f"{k} Percentile", f"{v:,}"] for k, v in stars_analysis['percentiles'].items()]
            print(tabulate(percentiles_table, headers=["Percentile", "Stars"], tablefmt="grid"))
        
        # Language distribution
        lang_analysis = self.analyze_language_distribution()
        if lang_analysis and 'language_stats' in lang_analysis:
            print("\n💻 TOP PROGRAMMING LANGUAGES")
            print("-" * 40)
            
            top_langs = lang_analysis['language_stats'][:10]
            lang_table = []
            for lang in top_langs:
                lang_table.append([
                    lang['language'],
                    f"{lang['repositories']:,}",
                    f"{lang['total_stars']:,}",
                    f"{lang['avg_stars']:,.0f}",
                    f"{lang['avg_contributors']:,.1f}"
                ])
            
            print(tabulate(lang_table, 
                         headers=["Language", "Repos", "Total Stars", "Avg Stars", "Avg Contributors"], 
                         tablefmt="grid"))
        
        # License distribution
        license_analysis = self.analyze_license_distribution()
        if license_analysis:
            print("\n📄 TOP LICENSES")
            print("-" * 40)
            
            license_table = []
            for license_name, count in list(license_analysis['top_licenses'].items())[:10]:
                percentage = license_analysis['license_percentages'][license_name]
                license_table.append([license_name, f"{count:,}", f"{percentage}%"])
            
            print(tabulate(license_table, headers=["License", "Repositories", "Percentage"], tablefmt="grid"))
        
        # Activity metrics
        activity_analysis = self.analyze_activity_metrics()
        if activity_analysis:
            print("\n🚀 ACTIVITY METRICS")
            print("-" * 40)
            
            if 'issues' in activity_analysis:
                issues = activity_analysis['issues']
                print(f"Issues - Avg: {issues['avg_total_issues']}, "
                      f"Median: {issues['median_total_issues']}, "
                      f"Repos with issues: {issues['percentage_with_issues']}%")
            
            if 'pull_requests' in activity_analysis:
                prs = activity_analysis['pull_requests']
                print(f"PRs - Avg: {prs['avg_total_prs']}, "
                      f"Median: {prs['median_total_prs']}, "
                      f"Repos with PRs: {prs['percentage_with_prs']}%")
            
            if 'contributors' in activity_analysis:
                contributors = activity_analysis['contributors']
                print(f"Contributors - Avg: {contributors['avg_contributors']}, "
                      f"Median: {contributors['median_contributors']}, "
                      f"Single contributor repos: {contributors['single_contributor_repos']}")
    
    def create_visualizations(self, save_plots: bool = True):
        """Create comprehensive visualizations"""
        if self.df is None:
            print("No data loaded. Please load data first.")
            return
        
        # Set up the plotting
        plt.rcParams['figure.figsize'] = (15, 10)
        
        # 1. Stars distribution
        fig, axes = plt.subplots(2, 2, figsize=(20, 15))
        fig.suptitle('GitHub Repository Analysis Dashboard', fontsize=20, fontweight='bold')
        
        # Stars histogram (log scale)
        axes[0, 0].hist(self.df['stars'], bins=50, alpha=0.7, color='skyblue', edgecolor='black')
        axes[0, 0].set_xscale('log')
        axes[0, 0].set_xlabel('Stars (log scale)')
        axes[0, 0].set_ylabel('Number of Repositories')
        axes[0, 0].set_title('Distribution of Stars (Log Scale)')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Language distribution (top 10)
        if 'language' in self.df.columns:
            top_languages = self.df['language'].value_counts().head(10)
            top_languages.plot(kind='bar', ax=axes[0, 1], color='lightcoral')
            axes[0, 1].set_xlabel('Programming Language')
            axes[0, 1].set_ylabel('Number of Repositories')
            axes[0, 1].set_title('Top 10 Programming Languages by Repository Count')
            axes[0, 1].tick_params(axis='x', rotation=45)
        
        # Stars vs Forks scatter plot
        if 'forks' in self.df.columns:
            scatter = axes[1, 0].scatter(self.df['stars'], self.df['forks'], 
                                       alpha=0.6, c=self.df['contributors'], 
                                       cmap='viridis', s=30)
            axes[1, 0].set_xlabel('Stars')
            axes[1, 0].set_ylabel('Forks')
            axes[1, 0].set_title('Stars vs Forks (colored by Contributors)')
            axes[1, 0].set_xscale('log')
            axes[1, 0].set_yscale('log')
            plt.colorbar(scatter, ax=axes[1, 0], label='Contributors')
        
        # License distribution (pie chart)
        if 'license' in self.df.columns:
            license_counts = self.df['license'].value_counts().head(8)
            other_count = self.df['license'].value_counts()[8:].sum()
            if other_count > 0:
                license_counts['Other'] = other_count
            
            axes[1, 1].pie(license_counts.values, labels=license_counts.index, autopct='%1.1f%%')
            axes[1, 1].set_title('License Distribution (Top 8 + Other)')
        
        plt.tight_layout()
        if save_plots:
            plt.savefig(self.data_path / 'github_analysis_dashboard.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # 2. Language-specific analysis
        if 'language' in self.df.columns:
            fig, axes = plt.subplots(2, 2, figsize=(20, 15))
            fig.suptitle('Programming Language Analysis', fontsize=20, fontweight='bold')
            
            # Average stars by language (top 10)
            lang_stars = self.df.groupby('language')['stars'].mean().sort_values(ascending=False).head(10)
            lang_stars.plot(kind='bar', ax=axes[0, 0], color='gold')
            axes[0, 0].set_xlabel('Programming Language')
            axes[0, 0].set_ylabel('Average Stars')
            axes[0, 0].set_title('Average Stars by Language (Top 10)')
            axes[0, 0].tick_params(axis='x', rotation=45)
            
            # Total issues by language (if available)
            if 'total_issues' in self.df.columns:
                lang_issues = self.df.groupby('language')['total_issues'].sum().sort_values(ascending=False).head(10)
                lang_issues.plot(kind='bar', ax=axes[0, 1], color='lightcoral')
                axes[0, 1].set_xlabel('Programming Language')
                axes[0, 1].set_ylabel('Total Issues')
                axes[0, 1].set_title('Total Issues by Language (Top 10)')
                axes[0, 1].tick_params(axis='x', rotation=45)
            
            # Contributors by language
            if 'contributors' in self.df.columns:
                lang_contributors = self.df.groupby('language')['contributors'].mean().sort_values(ascending=False).head(10)
                lang_contributors.plot(kind='bar', ax=axes[1, 0], color='lightgreen')
                axes[1, 0].set_xlabel('Programming Language')
                axes[1, 0].set_ylabel('Average Contributors')
                axes[1, 0].set_title('Average Contributors by Language (Top 10)')
                axes[1, 0].tick_params(axis='x', rotation=45)
            
            # CI/CD adoption by language (if available)
            if 'ci_cd' in self.df.columns:
                lang_ci_cd = self.df.groupby('language')['ci_cd'].mean().sort_values(ascending=False).head(10)
                lang_ci_cd.plot(kind='bar', ax=axes[1, 1], color='purple')
                axes[1, 1].set_xlabel('Programming Language')
                axes[1, 1].set_ylabel('CI/CD Adoption Rate')
                axes[1, 1].set_title('CI/CD Adoption by Language (Top 10)')
                axes[1, 1].tick_params(axis='x', rotation=45)
            
            plt.tight_layout()
            if save_plots:
                plt.savefig(self.data_path / 'language_analysis.png', dpi=300, bbox_inches='tight')
            plt.show()
        
        # 3. Additional metrics
        fig, axes = plt.subplots(2, 2, figsize=(20, 15))
        fig.suptitle('Repository Quality and Activity Metrics', fontsize=20, fontweight='bold')
        
        # Fork to star ratio distribution
        if 'fork_to_star_ratio' in self.df.columns:
            axes[0, 0].hist(self.df['fork_to_star_ratio'], bins=50, alpha=0.7, color='orange')
            axes[0, 0].set_xlabel('Fork to Star Ratio')
            axes[0, 0].set_ylabel('Number of Repositories')
            axes[0, 0].set_title('Distribution of Fork to Star Ratio')
            axes[0, 0].set_xlim(0, 1)  # Focus on reasonable range
        
        # README length distribution
        if 'readme_lines' in self.df.columns:
            readme_data = self.df[self.df['readme_lines'] > 0]['readme_lines']
            axes[0, 1].hist(readme_data, bins=50, alpha=0.7, color='lightblue')
            axes[0, 1].set_xlabel('README Lines')
            axes[0, 1].set_ylabel('Number of Repositories')
            axes[0, 1].set_title('Distribution of README Length')
        
        # Stars vs Contributors
        if 'contributors' in self.df.columns:
            axes[1, 0].scatter(self.df['contributors'], self.df['stars'], alpha=0.6, color='green')
            axes[1, 0].set_xlabel('Contributors')
            axes[1, 0].set_ylabel('Stars')
            axes[1, 0].set_title('Contributors vs Stars')
            axes[1, 0].set_yscale('log')
        
        # Repository features (wiki, pages, CI/CD)
        features_data = []
        feature_names = []
        
        if 'has_wiki' in self.df.columns:
            features_data.append(self.df['has_wiki'].sum())
            feature_names.append('Has Wiki')
        
        if 'has_pages' in self.df.columns:
            features_data.append(self.df['has_pages'].sum())
            feature_names.append('Has Pages')
        
        if 'ci_cd' in self.df.columns:
            features_data.append(self.df['ci_cd'].sum())
            feature_names.append('Has CI/CD')
        
        if features_data:
            axes[1, 1].bar(feature_names, features_data, color=['blue', 'red', 'green'][:len(features_data)])
            axes[1, 1].set_ylabel('Number of Repositories')
            axes[1, 1].set_title('Repository Features Adoption')
            axes[1, 1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        if save_plots:
            plt.savefig(self.data_path / 'quality_metrics.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def generate_full_report(self):
        """Generate a complete analysis report"""
        print("Generating comprehensive GitHub repository analysis...")
        
        if not self.load_data():
            print("Failed to load data. Please check the data files.")
            return
        
        # Print summary tables
        self.print_summary_tables()
        
        # Create visualizations
        print("\nGenerating visualizations...")
        self.create_visualizations()
        
        # Additional insights
        print("\n" + "=" * 80)
        print("📋 KEY INSIGHTS")
        print("=" * 80)
        
        if self.df is not None:
            # Top repositories by different metrics
            print("\n🌟 TOP 5 REPOSITORIES BY STARS:")
            top_by_stars = self.df.nlargest(5, 'stars')[['repo_name', 'owner', 'language', 'stars', 'forks']]
            print(tabulate(top_by_stars.values, 
                         headers=['Repository', 'Owner', 'Language', 'Stars', 'Forks'], 
                         tablefmt="grid"))
            
            if 'contributors' in self.df.columns:
                print("\n👥 TOP 5 REPOSITORIES BY CONTRIBUTORS:")
                top_by_contributors = self.df.nlargest(5, 'contributors')[['repo_name', 'owner', 'language', 'contributors', 'stars']]
                print(tabulate(top_by_contributors.values, 
                             headers=['Repository', 'Owner', 'Language', 'Contributors', 'Stars'], 
                             tablefmt="grid"))
            
            if 'fork_to_star_ratio' in self.df.columns:
                print("\n🍴 TOP 5 REPOSITORIES BY FORK-TO-STAR RATIO:")
                top_by_ratio = self.df.nlargest(5, 'fork_to_star_ratio')[['repo_name', 'owner', 'language', 'fork_to_star_ratio', 'stars', 'forks']]
                print(tabulate(top_by_ratio.values, 
                             headers=['Repository', 'Owner', 'Language', 'Fork/Star Ratio', 'Stars', 'Forks'], 
                             tablefmt="grid"))
        
        print(f"\n✅ Analysis complete! Charts saved to {self.data_path}")


def main():
    """Main function to run the analyzer"""
    # Initialize analyzer
    analyzer = GitHubDataAnalyzer()
    
    # Generate full report
    analyzer.generate_full_report()


if __name__ == "__main__":
    main()
