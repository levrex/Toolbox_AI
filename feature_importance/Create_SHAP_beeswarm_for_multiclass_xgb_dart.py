import shap
import time
import matplotlib.pyplot as plt

t0 = time.time()
# Train XGBoost model with booster='dart' (regular TreeExplainer does not work for dart)

# Wrapper function to safely handle DMatrix conversion -> For booster 'dart'
def safe_predict(data):
    return bst.predict(xgb.DMatrix(data))

# Use shap.Explainer with the wrapper
t_explainer = shap.Explainer(safe_predict, X_test)

#t_explainer = shap.Explainer(final_model.predict, x_test[l_elastic])
shap_values = t_explainer(X_test)
                                   

t1 = time.time()
print('Calculating SHAP: ' + str(t1-t0))


N_FEAT = 10

# Create SHAP per cluster
for i in range(4):
    plt.figure(figsize=(10, 6))  
    shap.plots.beeswarm(shap_values[:,:, i], max_display=N_FEAT, show=False)

    plt.savefig('shap_top%s_cluster_%s.png' % (str(N_FEAT), str(i)), dpi=100, bbox_inches='tight')
    # Save the figure
    plt.close()  # Close the figure to prevent displaying again
    plt.clf()