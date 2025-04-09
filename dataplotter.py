import matplotlib.pyplot as plt
import pandas as pd

# plot sensor data from a file
df = pd.read_csv("./data/Panu_trick.csv")

# Plot the sensor data
df.plot(
    x="Timestamp",
    y=["Sensor1", "Sensor2","Sensor3", "Sensor4"],
    title="Sensor data"
)
plt.xlabel("Timestamp (ms)")
plt.ylabel("Sensor value")
plt.grid(True)
plt.show()