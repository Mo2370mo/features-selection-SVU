import numpy as np
import warnings
import sys
import pandas as pd
import argparse
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE, SelectKBest, f_classif, SelectFromModel
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt

class GeneticFeatureSelector(BaseEstimator, TransformerMixin):

    def __init__(self, estimator, population_size=50, n_generations=20,
                 crossover_probability=0.8, mutation_probability=0.05,
                 tournament_size=3, alpha=0.9, cv=5, random_state=None,
                 verbose=1):
        self.estimator = estimator
        self.population_size = population_size
        self.n_generations = n_generations
        self.crossover_probability = crossover_probability
        self.mutation_probability = mutation_probability
        self.tournament_size = tournament_size
        self.alpha = alpha
        self.cv = cv
        self.random_state = random_state
        self.verbose = verbose

        self.rng = np.random.default_rng(self.random_state)
        self.population_ = []
        self.fitness_scores_ = []
        self.best_chromosome_ = None
        self.best_fitness_ = -np.inf
        self.generation_stats_ = []
        self.n_features_total_ = 0
        self.selected_features_indices_ = None
        self.n_features_selected_ = 0

    def _initialize_population(self):
        population = []
        for _ in range(self.population_size):
            chromosome = self.rng.integers(0, 2, size=self.n_features_total_)
            if np.sum(chromosome) == 0:
                chromosome[self.rng.integers(0, self.n_features_total_)] = 1
            population.append(chromosome)
        return population

    def _calculate_fitness(self, chromosome, X, y):
        num_features = np.sum(chromosome)
        if num_features == 0:
            return 0.0

        selected_indices = np.where(chromosome == 1)[0]
        X_subset = X[:, selected_indices]

        try:
            model = clone(self.estimator)
            scores = cross_val_score(model, X_subset, y, cv=self.cv, scoring='accuracy')
            performance = np.mean(scores)
        except ValueError:
            performance = 0.0

        feature_ratio = num_features / self.n_features_total_
        
        fitness = (self.alpha * performance) + ((1 - self.alpha) * (1 - feature_ratio))
        return fitness

    def _selection(self):
        tournament_indices = self.rng.choice(
            self.population_size, self.tournament_size, replace=False
        )
        tournament_fitnesses = [self.fitness_scores_[i] for i in tournament_indices]
        winner_index = tournament_indices[np.argmax(tournament_fitnesses)]
        
        return self.population_[winner_index]

    def _crossover(self, parent1, parent2):
        if self.rng.random() < self.crossover_probability:
            crossover_point = self.rng.integers(1, self.n_features_total_ - 1)
            child1 = np.concatenate((parent1[:crossover_point], parent2[crossover_point:]))
            child2 = np.concatenate((parent2[:crossover_point], parent1[crossover_point:]))
        else:
            child1, child2 = parent1.copy(), parent2.copy()
            
        return child1, child2

    def _mutation(self, chromosome):
        for i in range(self.n_features_total_):
            if self.rng.random() < self.mutation_probability:
                chromosome[i] = 1 - chromosome[i]
        
        if np.sum(chromosome) == 0:
            chromosome[self.rng.integers(0, self.n_features_total_)] = 1
            
        return chromosome

    def fit(self, X, y):
        self.n_features_total_ = X.shape[1]
        
        self.population_ = self._initialize_population()
        
        for gen in range(self.n_generations):
            self.fitness_scores_ = [self._calculate_fitness(chrom, X, y) for chrom in self.population_]

            current_best_idx = np.argmax(self.fitness_scores_)
            current_best_fitness = self.fitness_scores_[current_best_idx]

            if current_best_fitness > self.best_fitness_:
                self.best_fitness_ = current_best_fitness
                self.best_chromosome_ = self.population_[current_best_idx].copy()
            
            avg_fitness = np.mean(self.fitness_scores_)
            self.generation_stats_.append({
                'gen': gen, 
                'best_fitness': self.best_fitness_, 
                'avg_fitness': avg_fitness
            })
            if self.verbose > 0:
                print(f"الجيل {gen+1}/{self.n_generations} - أفضل لياقة: {self.best_fitness_:.4f}, متوسط اللياقة: {avg_fitness:.4f}, ميزات مختارة: {np.sum(self.best_chromosome_)}")

            new_population = []
            new_population.append(self.best_chromosome_.copy()) 

            while len(new_population) < self.population_size:
                parent1 = self._selection()
                parent2 = self._selection()
                
                child1, child2 = self._crossover(parent1, parent2)
                
                new_population.append(self._mutation(child1))
                if len(new_population) < self.population_size:
                    new_population.append(self._mutation(child2))
            
            self.population_ = new_population
        
        self.selected_features_indices_ = np.where(self.best_chromosome_ == 1)[0]
        self.n_features_selected_ = len(self.selected_features_indices_)
        
        return self

    def transform(self, X):
        if self.selected_features_indices_ is None:
            raise RuntimeError("لم يتم تدريب المختار بعد.")
        
        return X[:, self.selected_features_indices_]

def load_and_prepare_data(filepath, target_column):
    try:
        df = pd.read_csv(filepath)
    except FileNotFoundError:
        print(f"خطأ: الملف غير موجود في المسار {filepath}")
        return None, None
    except Exception as e:
        print(f"خطأ في تحميل CSV: {e}")
        return None, None

    if target_column not in df.columns:
        print(f"خطأ: عمود المتغير الهدف '{target_column}' غير موجود في الملف.")
        return None, None

    print(f"تم تحميل الملف '{filepath}' بنجاح.")
    y = df[target_column]
    X = df.drop(columns=[target_column])

    if y.dtype == 'object' or y.dtype.name == 'category':
        le = LabelEncoder()
        y = le.fit_transform(y)
        print("تم تطبيق تشفير (LabelEncoder) على المتغير الهدف لأنه فئوي.")
    
    numeric_features = X.select_dtypes(include=np.number).columns
    categorical_features = X.select_dtypes(exclude=np.number).columns

    if len(numeric_features) > 0:
        imputer_num = SimpleImputer(strategy='mean')
        X[numeric_features] = imputer_num.fit_transform(X[numeric_features])
        print("تم تعويض القيم المفقودة في الميزات الرقمية بالمتوسط.")

    if len(categorical_features) > 0:
        imputer_cat = SimpleImputer(strategy='most_frequent')
        X[categorical_features] = imputer_cat.fit_transform(X[categorical_features])
        
        X = pd.get_dummies(X, columns=categorical_features, drop_first=True)
        print("تم تطبيق تعويض (Most Frequent) وتشفير (One-Hot Encoding) للميزات الفئوية.")
    
    print(f"اكتملت المعالجة المسبقة. العدد النهائي للميزات: {X.shape[1]}")
    
    return X.values, y.values


def evaluate_model(X_train, X_test, y_train, y_test, estimator, method_name):
    n_features = X_train.shape[1]
    
    if n_features == 0:
        print(f"تحذير: {method_name} اختار 0 ميزة.")
        return 0.0, 0
    
    model = clone(estimator)
    try:
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        accuracy = accuracy_score(y_test, preds)
    except Exception as e:
        print(f"خطأ في تدريب/تقييم {method_name}: {e}")
        accuracy = 0.0
        
    return accuracy, n_features

def run_feature_selection(X_train, X_test, y_train, y_test, base_estimator, ga_n_features):
    results = {}
    print("\n--- 1. الأساس (Baseline): استخدام جميع الميزات ---")
    acc_all, n_feat_all = evaluate_model(X_train, X_test, y_train, y_test, base_estimator, "الأساس")
    results['Baseline'] = {'name': 'جميع الميزات', 'accuracy': acc_all, 'features': n_feat_all}
    print("\n--- 2. الإزالة التكرارية للميزات (RFE) ---")
    if ga_n_features < 1:
        n_features_rfe = 1 
    else:
        n_features_rfe = ga_n_features

    rfe_selector = RFE(
        estimator=clone(base_estimator),
        n_features_to_select=n_features_rfe,
        step=0.1
    )
    rfe_selector.fit(X_train, y_train)
    X_train_rfe = rfe_selector.transform(X_train)
    X_test_rfe = rfe_selector.transform(X_test)
    
    acc_rfe, n_feat_rfe = evaluate_model(X_train_rfe, X_test_rfe, y_train, y_test, base_estimator, "RFE")
    results['RFE'] = {'name': f'RFE (نفس عدد GA)', 'accuracy': acc_rfe, 'features': n_feat_rfe}

    print("\n--- 3. طريقة الترشيح (SelectKBest - ANOVA) ---")
  
    if ga_n_features < 1:
        n_features_filter = 1
    else:
        n_features_filter = ga_n_features
        
    filter_selector = SelectKBest(f_classif, k=n_features_filter)
    filter_selector.fit(X_train, y_train)
    X_train_filter = filter_selector.transform(X_train)
    X_test_filter = filter_selector.transform(X_test)
    
    acc_filter, n_feat_filter = evaluate_model(X_train_filter, X_test_filter, y_train, y_test, base_estimator, "Filter")
    results['Filter'] = {'name': f'ترشيح (نفس عدد GA)', 'accuracy': acc_filter, 'features': n_feat_filter}


    print("\n--- 4. طريقة التضمين (Lasso/L1 Regularization) ---")
    l1_model = LogisticRegression(
        penalty='l1', 
        C=0.1, 
        solver='liblinear',
        random_state=42
    )
    
    embedded_selector = SelectFromModel(l1_model, prefit=False)
    embedded_selector.fit(X_train, y_train)
    
    X_train_embedded = embedded_selector.transform(X_train)
    X_test_embedded = embedded_selector.transform(X_test)
    
    acc_embedded, n_feat_embedded = evaluate_model(X_train_embedded, X_test_embedded, y_train, y_test, base_estimator, "Embedded (Lasso)")
    results['Embedded'] = {'name': 'تضمين (Lasso L1)', 'accuracy': acc_embedded, 'features': n_feat_embedded}
    
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Genetic Algorithm Feature Selector and Comparison Tool",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        'csv_file', 
        type=str, 
        nargs='?', 
        help="Path to the input CSV file."
    )
    parser.add_argument(
        'target_column', 
        type=str, 
        nargs='?', 
        help="Name of the target variable column in the CSV."
    )
    parser.add_argument('--pop_size', type=int, default=20, help="Population size for GA (default: 20)")
    parser.add_argument('--generations', type=int, default=10, help="Number of generations for GA (default: 10)")
    parser.add_argument('--alpha', type=float, default=0.9, help="Weight for performance in GA fitness (default: 0.9)")
    
    args = parser.parse_args()

    if not args.csv_file or not args.target_column:
        from sklearn.datasets import make_classification
        print("\n--- لم يتم تزويد ملف. إنشاء ملف 'sample_data.csv' تجريبي للمقارنة. ---")
        
        X_dummy, y_dummy = make_classification(
            n_samples=200, n_features=30, n_informative=8, n_redundant=2,
            n_useless=20, random_state=42
        )
        dummy_df = pd.DataFrame(X_dummy, columns=[f'feature_{i}' for i in range(30)])
        dummy_df['target'] = y_dummy
        dummy_df.to_csv('sample_data.csv', index=False)
        
        args.csv_file = 'sample_data.csv'
        args.target_column = 'target'
        args.pop_size = 15
        args.generations = 5
        print(f"الرجاء تشغيل الكود بملف البيانات الخاص بك: python {sys.argv[0]} your_file.csv your_target\n")

    print("\n--- بدء أداة اختيار الميزات والمقارنة الشاملة ---")

    X, y = load_and_prepare_data(args.csv_file, args.target_column)
    
    if X is None or y is None:
        sys.exit(1)

    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
        )
    except ValueError:
        print("تحذير: لا يمكن تقسيم البيانات بشكل طبقي. تم استخدام تقسيم عشوائي عادي.")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42
        )
    
    base_estimator = RandomForestClassifier(n_estimators=10, random_state=42)
    print("\n--- أ. تشغيل خوارزمية الاختيار الجينية (GA) ---")
    
    ga_selector = GeneticFeatureSelector(
        estimator=clone(base_estimator),
        population_size=args.pop_size,
        n_generations=args.generations,
        alpha=args.alpha,
        cv=3, 
        random_state=42,
        verbose=1
    )

    ga_selector.fit(X_train, y_train)

    X_train_ga = ga_selector.transform(X_train)
    X_test_ga = ga_selector.transform(X_test)
    
    acc_ga, n_feat_ga = evaluate_model(X_train_ga, X_test_ga, y_train, y_test, base_estimator, "GA")
    
    comparison_results = {
        'GA': {'name': 'الخوارزمية الجينية (GA)', 'accuracy': acc_ga, 'features': n_feat_ga}
    }
    
    if n_feat_ga > 0:
        comparison_results.update(
            run_feature_selection(X_train, X_test, y_train, y_test, base_estimator, n_feat_ga)
        )
    else:
        print("\n**تنبيه: الخوارزمية الجينية اختارت 0 ميزة. لا يمكن إجراء مقارنة عادلة.**")
        comparison_results.update({
            'Baseline': {'name': 'جميع الميزات', 'accuracy': 0.0, 'features': X_train.shape[1]},
            'RFE': {'name': 'RFE', 'accuracy': 0.0, 'features': 0},
            'Filter': {'name': 'ترشيح', 'accuracy': 0.0, 'features': 0},
            'Embedded': {'name': 'تضمين (Lasso L1)', 'accuracy': 0.0, 'features': 0},
        })
    
    print("\n" + "="*80)
    print("نتائج المقارنة الشاملة لطرق اختيار الميزات".center(80))
    print("="*80)
    
    display_order = ['Baseline', 'GA', 'RFE', 'Filter', 'Embedded']
    
    header = f"| {'الطريقة':<30} | {'دقة النموذج (Accuracy)':<25} | {'عدد الميزات المستخدمة':<20} |"
    print(header)
    print("-" * 80)

    for key in display_order:
        result = comparison_results.get(key)
        if result:
            row = (
                f"| {result['name']:<30} | "
                f"{result['accuracy'] * 100:.2f}% (على بيانات الاختبار)  {'' if result['accuracy'] == max(r['accuracy'] for r in comparison_results.values() if r) else '':'<25'} | "
                f"{result['features']:<20} |"
            )
            print(row)
        
    print("="*80)
    
    stats = ga_selector.generation_stats_
    
    if stats:
        plt.figure(figsize=(10, 6))
        plt.plot([s['gen'] for s in stats], [s['best_fitness'] for s in stats], 
                 label='أفضل لياقة (Best Fitness)', marker='o', linestyle='-')
        plt.plot([s['gen'] for s in stats], [s['avg_fitness'] for s in stats], 
                 label='متوسط اللياقة (Average Fitness)', marker='x', linestyle='--')
        plt.title('تطور لياقة الخوارزمية الجينية عبر الأجيال')
        plt.xlabel('الجيل')
        plt.ylabel('درجة اللياقة')
        plt.legend()
        plt.grid(True)
        print("\nيُرجى إغلاق نافذة الرسم البياني لعرض النتائج الكاملة إذا لم تظهر بعد.")
        plt.show()