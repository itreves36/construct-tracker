# Create a variable time of day but in radians

# time_of_day_radian


import matplotlib.pyplot as plt
import numpy as np


# Define the conversion from hours to radians
def time_to_cyclical_cos_neg(time_of_day):
    return np.cos(time_of_day * (2 * np.pi / 24)) * (-1)


# Define the conversion from hours to radians for sine
def time_to_cyclical_sine(time_of_day):
    return np.sin(time_of_day * (2 * np.pi / 24))


# Calculate cyclical time values for example times
times_of_day = range(1, 25)  # Example times: Midnight, 6AM, Noon, 6PM, 11PM
cyclical_values_cos = [time_to_cyclical_cos_neg(time) for time in times_of_day]

# Calculate cyclical sine and cos values for example times
cyclical_values_sine = [time_to_cyclical_sine(time) for time in times_of_day]

cmap = plt.get_cmap("viridis_r")  # You can choose any available colormap

# Generate n colors from the colormap
colors = cmap(np.linspace(0, 1, 24))

# Pair each time with its cyclical sine and cos values
time_cyclical_pairs_sine_cos = list(zip(times_of_day, cyclical_values_cos, cyclical_values_sine, colors))
time_cyclical_pairs_sine_cos


for hour, cyclical_value, cyclical_value_sine, color in time_cyclical_pairs_sine_cos:
    plt.scatter(cyclical_value, cyclical_value_sine, marker="o", label=hour, color=color)
plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))
plt.xlabel("cos(hour*2*pi/24)")
plt.ylabel("sin(hour*2*pi/24)")
plt.tight_layout()

plt.show()

# for hour, cyclical_value, cyclical_value_sine, color in time_cyclical_pairs_sine_cos:
# 	plt.scatter(0,cyclical_value, marker='o', label = hour, color = color)
# # plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
# plt.tight_layout()

# plt.show()


for hour, cyclical_value, _cyclical_value_sine, color in time_cyclical_pairs_sine_cos:
    plt.scatter(hour, cyclical_value, marker="o", label=hour, color=color)
# plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.ylabel("cos(hour*2*pi/24)*(-1)\n Approximation for Daylight")
plt.xlabel("Hour")
plt.tight_layout()
plt.show()

for hour, _cyclical_value, cyclical_value_sine, color in time_cyclical_pairs_sine_cos:
    plt.scatter(hour, cyclical_value_sine, marker="o", label=hour, color=color)
# plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.ylabel("sin(hour*2*pi/24)")
plt.xlabel("Hour")
plt.tight_layout()
plt.show()


for hour, cyclical_value, cyclical_value_sine, color in time_cyclical_pairs_sine_cos:
    plt.scatter(hour, cyclical_value_sine * cyclical_value, marker="o", label=hour, color=color)
plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))
plt.tight_layout()

plt.show()


for hour, cyclical_value, cyclical_value_sine, color in time_cyclical_pairs_sine_cos:
    plt.scatter(hour, hour * cyclical_value_sine * cyclical_value, marker="o", label=hour, color=color)
plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))
plt.tight_layout()

plt.show()


"""
Example:
# Obtain time of day from timestamp str

X_df['time_of_day'] = X_df['datetime_str_new_york_voice'].dt.hour.astype(str).astype(int)

X_df['time_of_day_cos_daylight'] = X_df['time_of_day'].apply(time_to_cyclical_cos_neg)

for i, idx in enumerate(X_df['participant_id'].unique()[:10]):
	radians = X_df[X_df['participant_id'] == idx]['time_of_day_cos_daylight'].values
	time_of_day = X_df[X_df['participant_id'] == idx]['time_of_day'].values
	# timestamp = X_df[X_df['participant_id'] == idx]['datetime_str_new_york_voice'].values

	# time_of_day = [n.split(' ')[-1].split(':')[0] for n in timestamp]
	plt.scatter(time_of_day, radians, marker='o', label = idx, color = colors[i])
	plt.xlim(-1, 25)
	plt.ylim(-1.1, 1.1)
	plt.show()


"""
