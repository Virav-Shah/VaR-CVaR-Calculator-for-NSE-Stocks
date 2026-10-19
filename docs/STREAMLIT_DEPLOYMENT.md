# Keep the existing Streamlit URL

The GitHub repository has been restored to its original name so the existing Streamlit deployment can find the coordinates it was created with. All project improvements remain on the same repository and main branch.

- Existing application: https://mkxk2evmghkv5rbpugw99x.streamlit.app/
- Repository: **Virav-Shah/VaR-CVaR-Calculator-for-NSE-Stocks**
- Branch: **main**
- Entry point: **app.py**

## Refresh the existing deployment

1. Sign into the Streamlit Cloud account that owns the existing app.
2. Open the existing app and choose **Manage app**.
3. In the management panel, choose **⋮ → Reboot app**, then confirm.
4. Wait for the build to finish and refresh the same application URL.

Keep the existing app rather than deleting it or creating a replacement. A reboot rebuilds the current deployment and retains its URL. See [Streamlit reboot instructions](https://docs.streamlit.io/deploy/streamlit-community-cloud/manage-your-app/reboot-your-app).

If app management is unavailable after restoring the repository name, sign out of both Streamlit Community Cloud and GitHub, then sign back in. Restoring the original GitHub coordinates and signing in again is the recovery sequence documented by [Streamlit](https://docs.streamlit.io/deploy/streamlit-community-cloud/manage-your-app/rename-your-app). If access is still unavailable, contact Streamlit support about recovering the existing deployment and retaining its URL.

## Recognize the current version

- Title: **NSE Market Risk Analytics**.
- Five tabs: **EDA & Regimes**, **VaR / CVaR**, **Regulatory Backtesting**, **Explainable AI**, **Portfolio Optimization**.
- No ML Lifecycle tab.
- Both V1 and V2 explainability use saved contributions; app.py does not import the SHAP library.
- Offline model training is unnecessary: required forecasts and explanation data are included in results/dashboard_v2/.

If you still see **NSE ML Risk & Governance Workbench** or **ML Lifecycle**, the deployment is serving an older revision. The application fixes were published in commit 85d203f and remain in subsequent commits.
