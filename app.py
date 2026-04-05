from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
import pickle

app = Flask(__name__)

# --- Load model and label encoder ---
model = pickle.load(open('crop_model.pkl', 'rb'))
le = pickle.load(open('label_encoder.pkl', 'rb'))

# --- Load popularity dataset ---
pop_df = pd.read_csv('Dataset_1_1.csv')

# --- Season map ---
crop_season_map = {
    "rice": ["kharif"], "jowar": ["kharif", "rabi"], "bajara": ["kharif"],
    "maize": ["kharif"], "ragi": ["kharif"], "cotton": ["kharif"],
    "soyabean": ["kharif"], "groundnut": ["kharif"], "sesame": ["kharif"],
    "toor": ["kharif"], "wheat": ["rabi"], "chickpea": ["rabi"],
    "mustard": ["rabi"], "sugarcane": ["perennial"], "sunflower": ["kharif", "rabi"],
    "castor": ["kharif", "rabi"]
}

def month_to_season(month):
    if isinstance(month, int):
        if month in [6, 7, 8, 9, 10]: return "kharif"
        elif month in [11, 12, 1, 2, 3]: return "rabi"
        else: return "zaid"
    m = str(month).strip().lower()
    kharif = {"june","jul","july","august","aug","september","sep","october","oct"}
    rabi = {"november","nov","december","dec","january","jan","february","feb","march","mar"}
    if m in kharif: return "kharif"
    elif m in rabi: return "rabi"
    else: return "zaid"

def popularity_score(rank, max_rank):
    if max_rank <= 1: return 1.0
    return 1 - (rank - 1) / (max_rank - 1)

def get_recommendations(n, p, k, ph, rainfall, district, month, top_k=5, alpha=0.7, beta=0.3):
    # Step 1: ML prediction
    x = pd.DataFrame([[n, p, k, ph, rainfall]], columns=["n", "p", "k", "ph", "rainfall"])
    probs = model.predict_proba(x)[0]
    probs = (probs + 1e-6) / (probs + 1e-6).sum()
    top_idx = np.argsort(probs)[-10:][::-1]
    crops = le.inverse_transform(top_idx)
    ml_df = pd.DataFrame({"crop": crops, "suitability": probs[top_idx]})

    # Step 2: Merge with popularity
    district_df = pop_df[pop_df["district"] == district][["crop", "popularity_rank"]]
    merged = ml_df.merge(district_df, on="crop", how="inner")

    if merged.empty:
        result = ml_df.head(top_k)
        return result[["crop", "suitability"]].to_dict(orient="records")

    # Step 3: Compute final score
    max_rank = merged["popularity_rank"].max()
    merged["popularity_score"] = merged["popularity_rank"].apply(lambda r: popularity_score(r, max_rank))
    merged["final_score"] = alpha * merged["suitability"] + beta * merged["popularity_score"]

    # Step 4: Season filter
    season = month_to_season(month)
    merged["allowed"] = merged["crop"].apply(
        lambda c: "perennial" in crop_season_map.get(c, []) or season in crop_season_map.get(c, [])
    )
    merged = merged[merged["allowed"] == True].drop(columns=["allowed"])

    # Step 5: Return top-k
    result = merged.sort_values("final_score", ascending=False).head(top_k)
    return result[["crop", "suitability", "popularity_score", "final_score"]].to_dict(orient="records")


@app.route('/recommend', methods=['POST'])
def recommend():
    try:
        data = request.get_json()
        n        = float(data['n'])
        p        = float(data['p'])
        k        = float(data['k'])
        ph       = float(data['ph'])
        rainfall = float(data['rainfall'])
        district = str(data['district'])
        month    = data['month']
        top_k    = int(data.get('top_k', 5))

        # ADD THIS
        if n == 0 and p == 0 and k == 0 and ph == 0:
            return jsonify({"status": "error", "message": "Invalid soil readings. Please connect the sensor or enter values manually."}), 400

        results = get_recommendations(n, p, k, ph, rainfall, district, month, top_k)
        return jsonify({"status": "success", "recommendations": results})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 400


if __name__ == '__main__':
    app.run(debug=True)
    
