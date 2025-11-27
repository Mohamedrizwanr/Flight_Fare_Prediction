from flask import Flask, render_template, request
from flask_cors import cross_origin
import pandas as pd
import pickle
from datetime import datetime, timedelta
import json

app = Flask(__name__)
model = pickle.load(open("flight_rf.pkl", "rb"))

# Airline order (same as your training dataset)
airline_order = [
    'Air India', 'GoAir', 'IndiGo', 'Jet Airways', 'Jet Airways Business',
    'Multiple carriers', 'Multiple carriers Premium economy',
    'SpiceJet', 'Trujet', 'Vistara', 'Vistara Premium economy'
]


@app.route("/")
@cross_origin()
def home():
    return render_template("home.html")


@app.route("/compare", methods=["POST"])
@cross_origin()
def compare():
    try:
        # -------- User Inputs --------
        date_dep = request.form["Dep_Time"]
        date_arr = request.form["Arrival_Time"]
        Journey_day = int(pd.to_datetime(date_dep).day)
        Journey_month = int(pd.to_datetime(date_dep).month)

        Dep_hour = int(pd.to_datetime(date_dep).hour)
        Dep_min = int(pd.to_datetime(date_dep).minute)
        Arrival_hour = int(pd.to_datetime(date_arr).hour)
        Arrival_min = int(pd.to_datetime(date_arr).minute)

        dur_hour = abs(Arrival_hour - Dep_hour)
        dur_min = abs(Arrival_min - Dep_min)

        Total_stops = int(request.form["stops"])
        min_price = int(request.form["min_price"])
        max_price = int(request.form["max_price"])

        # Source encoding
        Source = request.form["Source"]
        s_Chennai = 1 if Source == 'Chennai' else 0
        s_Delhi = 1 if Source == 'Delhi' else 0
        s_Kolkata = 1 if Source == 'Kolkata' else 0
        s_Mumbai = 1 if Source == 'Mumbai' else 0

        # Destination encoding
        Destination = request.form["Destination"]
        d_Cochin = 1 if Destination == 'Cochin' else 0
        d_Delhi = 1 if Destination == 'Delhi' else 0
        d_Hyderabad = 1 if Destination == 'Hyderabad' else 0
        d_Kolkata = 1 if Destination == 'Kolkata' else 0
        d_New_Delhi = 1 if Destination == 'Chennai' else 0

        # -------- Prediction Helper Function --------
        def predict_fares_for_date(day, month):
            fares = []
            for airline in airline_order:

                airline_features = [1 if airline == a else 0 for a in airline_order]

                row = [
                    Total_stops, day, month,
                    Dep_hour, Dep_min, Arrival_hour, Arrival_min,
                    dur_hour, dur_min
                ] + airline_features + [
                    s_Chennai, s_Delhi, s_Kolkata, s_Mumbai,
                    d_Cochin, d_Delhi, d_Hyderabad, d_Kolkata, d_New_Delhi
                ]

                price = model.predict([row])[0]
                fares.append(price)

            return fares

        # -------- Prediction for Selected Date --------
        today_fares = predict_fares_for_date(Journey_day, Journey_month)
        df_today = pd.DataFrame({
            "Airline": airline_order,
            "Predicted Fare (₹)": [round(f, 2) for f in today_fares]
        })

        # -------- Fare Status Logic --------
        def label_range(price):
            if price < min_price:
                return "Very Cheap"
            elif price > max_price:
                return "Expensive"
            else:
                return "Meets Expectation"

        df_today["Fare Status"] = df_today["Predicted Fare (₹)"].apply(label_range)

        df_today = df_today.sort_values("Predicted Fare (₹)")

        best_airline = df_today.iloc[0]["Airline"]
        best_price = df_today.iloc[0]["Predicted Fare (₹)"]

        # -------- Predict Next 7 Days Trend --------
        dep_date = datetime.strptime(date_dep, "%Y-%m-%dT%H:%M")
        trend_data = []
        calendar_data = []

        for i in range(7):
            next_date = dep_date + timedelta(days=i)
            fares = predict_fares_for_date(next_date.day, next_date.month)
            min_fare = min(fares)
            best_airline_next = airline_order[fares.index(min_fare)]

            trend_data.append({
                "date": next_date.strftime("%d-%b"),
                "fare": round(min_fare, 2)
            })
            calendar_data.append({
                "date": next_date.strftime("%d-%b"),
                "fare": round(min_fare, 2),
                "airline": best_airline_next
            })

        # -------- Cheapest Day --------
        cheapest_day = min(calendar_data, key=lambda x: x["fare"])
        cheapest_info = f" Cheapest day: {cheapest_day['date']} — ₹{cheapest_day['fare']} ({cheapest_day['airline']})"

        # -------- Status Message --------
        has_flight_in_range = (df_today["Fare Status"] == "Meets Expectation").any()

        status_msg = (
            f" Flights available within your expected range on {Journey_day}-{Journey_month}"
            if has_flight_in_range else
            f"⚠️ No flights matched your expected price range on {Journey_day}-{Journey_month}"
        )

        # -------- Render Output --------
        return render_template(
            "home.html",
            best_text=f"Best Fare Today: {best_airline} — ₹{best_price}",
            status_text=status_msg,
            result_table=df_today.to_html(classes="table table-striped table-bordered text-center", index=False),
            trend_data=json.dumps(trend_data),
            calendar_data=json.dumps(calendar_data),
            cheapest_text=cheapest_info
        )

    except Exception as e:
        return render_template("home.html", result_table=f"<p style='color:red;'>Error: {e}</p>")


if __name__ == "__main__":
    app.run(debug=True)
