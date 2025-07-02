from src.data_loader import load_tracking_data, load_plays
from src.preprocess import prepare_data
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import os

if __name__ == "__main__":
    print("🔄 Carregando dados...")
    tracking_df = load_tracking_data(week=1)
    plays_df = load_plays()

    print("⚙️  Preparando dados...")
    X, y = prepare_data(tracking_df, plays_df)

    print(f"✅ Dados prontos: {X.shape[0]} amostras, {X.shape[1]} features")

    print("🔀 Dividindo em treino/teste...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print("📈 Treinando modelo (Gradient Boosting)...")
    model = GradientBoostingRegressor(random_state=42)
    model.fit(X_train, y_train)

    print("📊 Avaliando modelo...")
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"🔍 MSE: {mse:.2f}")
    print(f"🔍 R²: {r2:.2f}")

    print("💾 Salvando modelo...")
    os.makedirs("models", exist_ok=True)
    joblib.dump(model, "models/trained_model.pkl")

    print("✅ Treinamento finalizado com sucesso.")
