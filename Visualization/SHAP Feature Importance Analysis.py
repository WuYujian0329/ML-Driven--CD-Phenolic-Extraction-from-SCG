shap_values_polyphenol = explainer_polyphenol.shap_values(X)

explainer_chlorogenic = shap.TreeExplainer(dt_model_chlorogenic) 
shap_values_chlorogenic = explainer_chlorogenic.shap_values(X)

shap.summary_plot(shap_values_polyphenol, X, show=False)
plt.title("Polyphenol Content SHAP Feature Importance Analysis")
plt.show()

shap.summary_plot(shap_values_chlorogenic, X, show=False)
plt.title("Chlorogenic Acid SHAP Feature Importance Analysis")
plt.show()
