# Deploy the current dashboard after the repository rename

Streamlit Community Cloud identifies a deployment by its GitHub owner, repository, branch and entry point. Renaming the GitHub repository can leave an existing deployment disconnected from future updates. See [Streamlit's official instructions](https://docs.streamlit.io/deploy/streamlit-community-cloud/manage-your-app/rename-your-app).

## Create a deployment from the new repository

1. Open your workspace at https://share.streamlit.io/.
2. Choose **Create app**, then deploy from GitHub.
3. Set repository to **Virav-Shah/nse-market-risk-analytics**.
4. Set branch to **main**.
5. Set the main file path to **app.py**.
6. Deploy and open the URL for that new deployment.

The existing public app may still serve an older revision. A browser refresh or reconnecting your GitHub account does not change its repository coordinates. If keeping the same app subdomain is necessary, follow Streamlit's documented delete/rename/redeploy flow. Preserve any app settings or secrets before deleting an old deployment. This dashboard does not require secrets.

## Recognize the current version

- Title: **NSE Market Risk Analytics**.
- Five tabs: **EDA & Regimes**, **VaR / CVaR**, **Regulatory Backtesting**, **Explainable AI**, **Portfolio Optimization**.
- No ML Lifecycle tab.
- Both V1 and V2 explainability use saved contributions; `app.py` does not import the SHAP library.
- Offline model training is unnecessary: required forecast and explanation data are included in `results/dashboard_v2/`.

If you still see **NSE ML Risk & Governance Workbench** or **ML Lifecycle**, the browser is displaying an older deployment. Check the new deployment's GitHub coordinates and open its own URL. The relevant application changes were published in commit `85d203f` and remain in subsequent commits.
