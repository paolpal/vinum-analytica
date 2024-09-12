from scipy.stats import wilcoxon

def wilcoxon_test(best_index, model_data):
    best_accuracies = model_data[best_index]['accuracies']
    results = []
    for i, data in enumerate(model_data):
        if i != best_index:
            result = wilcoxon(best_accuracies, data['accuracies'])
            results.append({
                'model': data.get('model_name', f'{i}'),
                'p_value': result.pvalue
            })
    return results