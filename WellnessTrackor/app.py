# -- coding:utf-8 --

# Author YI LIAO(Steven)
from flask import Flask, render_template, request, redirect, url_for, jsonify,render_template_string
from datetime import datetime, timedelta
from collections import defaultdict
import pymysql

from WearableDeviceData.wearableDeviceData import wearable_device_data
app=Flask(__name__)
activity_list=[]
syncdata_list=[]

def is_same_week(date1: datetime.date, date2: datetime.date) -> bool:
    return date1.isocalendar()[:2] == date2.isocalendar()[:2]

def insert_data_into_database(value_list):
    connection = pymysql.connect(host='localhost',
                                 user='root',
                                 passwd='root123',
                                 charset='utf8',
                                 database="wellness")
    print("connect database successfully")
    cursor = connection.cursor(cursor=pymysql.cursors.DictCursor)
    cursor.execute("insert into userinfo values(NULL, %s,%s,%s,%s,%s,%s);", value_list)
    connection.commit()

    cursor.close()
    connection.close()

def inquire_data_from_database(user_id, start_date, end_date):
    connection = pymysql.connect(host='localhost',
                                 user='root',
                                 passwd='root123',
                                 charset='utf8',
                                 database="wellness"
                                 )

    cursor = connection.cursor(cursor=pymysql.cursors.DictCursor)
    print("connect database successfully")
    sql_comand="select * from userinfo where user_id = %s AND date>=%s AND date<=%s;"
    cursor.execute(sql_comand, (user_id,start_date,end_date))
    data_list=cursor.fetchall()
    #print(data_list)
    cursor.close()
    connection.close()
    return data_list

def acquire_data_from_database(user_id):
    connection = pymysql.connect(host='localhost',
                                 user='root',
                                 passwd='root123',
                                 charset='utf8',
                                 database="wellness"
                                 )

    cursor = connection.cursor(cursor=pymysql.cursors.DictCursor)
    print("connect database successfully")
    sql_comand = "select * from userinfo where user_id = %s"
    cursor.execute(sql_comand, (user_id))
    data_list = cursor.fetchall()
    print(data_list)
    cursor.close()
    connection.close()
    return data_list


@app.route('/')
def index():  # put application's code here
    return render_template("WellnessTrack.html")

@app.route('/log-wellness-activity', methods=['POST'])
def log_wellness_activity():
    data = request.get_json()
    activity = {
        "user_id": data.get("user_id"),
        "date": data.get("date"),
        "hydration_liters": data.get("hydration_liters"),
        "sleep_hours": data.get("sleep_hours"),
        "exercise_minutes":data.get("exercise_minutes"),
        "meditation_minutes":data.get("meditation_minutes")
    }
    value_list = [activity["user_id"],
                  activity["date"],
                  activity["hydration_liters"],
                  activity["sleep_hours"],
                  activity["exercise_minutes"],
                  activity["meditation_minutes"]
                  ]
    insert_data_into_database(value_list)
    print(activity["user_id"])
    activity_list.append(activity)
    return jsonify({"message":"activity has been recorded into database successfully", "activity": activity}), 201


@app.route('/device-activity', methods=['GET'])
def device_activity():
    return jsonify([
        {"user_id": "running", "duration": 25, "timestamp": datetime.utcnow().isoformat()},
        {"type": "hydration", "duration": 0, "timestamp": datetime.utcnow().isoformat()}
    ]), 200

@app.route('/sync-device', methods=['POST'])
def sync_device():
    mock_data = wearable_device_data;
    num_activities=len(mock_data)
    for activity in mock_data:
        syncdata_list.append(activity)
        value_list = [activity["user_id"],
                      activity["date"],
                      activity["hydration_liters"],
                      activity["sleep_hours"],
                      activity["exercise_minutes"],
                      activity["meditation_minutes"]
                      ]
        insert_data_into_database(value_list)

    return jsonify({"data number":num_activities, "message":"wearable device data has been synced into database successfully!"}), 200

@app.route('/search')
def search_page():
    return render_template("Retrieval.html")
@app.route('/summary')
def summary_page():
    return render_template("Summary.html")

@app.route('/history', methods=['POST'])
def history():
    data=request.get_json()
    user_id=data.get('user_id')
    start = data.get('start_date')
    end   = data.get('end_date')
    print("user id:", user_id)
    data_list=inquire_data_from_database(user_id,start,end) #[{},{},,...]
    #return render_template("retrievalResult.html", data_list=data_list)
    return jsonify(data_list), 200


@app.route('/summary',methods=['POST'])
def summary():
    data = request.get_json()
    user_id=data.get('user_id')
    period=data.get('period')
    data_list=acquire_data_from_database(user_id)
    if period=="daily":
        num_instances=len(data_list)
        hydration=0.0
        sleep=0.0
        exercise=0.0
        meditation=0.0
        for item_dict in data_list:
            hydration=hydration+item_dict["hydration_liters"]
            sleep=sleep+item_dict["sleep_hours"]
            exercise=exercise+item_dict["exercise_minutes"]
            meditation=meditation+item_dict["meditation_minutes"]
        output={"user_id":item_dict["user_id"],
                "daily hydration liters":hydration/num_instances,
                "daily sleep hours":sleep/num_instances,
                "daily exercise minutes":exercise/num_instances,
                "daily meditation minutes": meditation/num_instances}
        print(output)

    else:
        d={user_id:period}

        week_hydaration_list =[data_list[0]["hydration_liters"]]
        week_sleep_list =[data_list[0]['sleep_hours']]
        week_exercise_list = [data_list[0]['exercise_minutes']]
        week_meditation_list = [data_list[0]["meditation_minutes"]]
        j=0

        for i in range(len(data_list)):
            if (i<len(data_list)-1) and (is_same_week(data_list[i]["date"], data_list[i+1]['date'])):
                j = j

                week_hydaration_list[j]=week_hydaration_list[j]+data_list[i+1]["hydration_liters"]
                week_sleep_list[j]=week_sleep_list[j]+data_list[i+1]['sleep_hours']
                week_exercise_list[j]=week_exercise_list[j]+data_list[i+1]['exercise_minutes']
                week_meditation_list[j]=week_meditation_list[j]+data_list[i+1]["meditation_minutes"]

            else:
                j=j+1
                week_hydaration_list.append(data_list[i]["hydration_liters"])
                week_sleep_list.append(data_list[i]['sleep_hours'])
                week_exercise_list.append(data_list[i]['exercise_minutes'])
                week_meditation_list.append(data_list[i]["meditation_minutes"])

        hydration =sum(week_hydaration_list)/len(week_hydaration_list)
        sleep =sum(week_sleep_list)/len(week_sleep_list)
        exercise = sum(week_exercise_list)/len(week_exercise_list)
        meditation = sum(week_meditation_list)/len(week_meditation_list)
        output = {"user_id": user_id,
                  "weekly hydration liters": hydration,
                  "weekly sleep hours": sleep ,
                  "weekly exercise minutes": exercise ,
                  "weekly meditation minutes": meditation}


    return jsonify(output), 200


if __name__ == "__main__":


   app.run(debug=True)



