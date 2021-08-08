# preprocessing
def convert_to_cat(cat_col_list, data):
	# convert object Dtype to categorial data
	for col in cat_col_list:
		data[col] = data[col].astype('category')
	return data




def median_age_by_occupation(gender, data):
	# sex: Male or Female
	is_male = data['sex'] == gender
	age_occupation = data[["sex", "age", "occupation"]][is_male]
	return age_occupation.groupby("occupation").median("age")




# plot
def histo(ax, X, x_label=None, y_label=None,color='#FEE08F'):
    n, bins, bpatches = ax.hist(X, color=color)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)

    # make it look good
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(.5)
    ax.spines['bottom'].set_linewidth(.5)
    for rect in bpatches:
        rect.set_linewidth(.5)
        rect.set_edgecolor('grey')
    return n, bins, bpatches 



def barplot(ax, x, y):
	barcontainers = ax.bar(x, y, color='#FEE08F')
	ax.spines['top'].set_visible(False)
	ax.spines['right'].set_visible(False)
	ax.spines['left'].set_linewidth(.5)
	ax.spines['bottom'].set_linewidth(.5)
	ax.set_ylabel("Avg mpg")

	for rect in barcontainers.patches:
		rect.set_linewidth(.5)
		rect.set_edgecolor('grey')