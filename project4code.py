# imported libraries
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.tree import export_graphviz
from pydotplus import graph_from_dot_data
from sklearn import preprocessing

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

# Read data from file
READ_FILE = 'Cumulative.csv'

# Headers for our data
HEADERS = ["ID", #1 
          "Sex", #2
          "Race", #3
          "First Generation",  #4
          "SAT Reading Score", #5
          "SAT Math Score",    #6
          "High School GPA",   #7
          "Semesters Taken To Graduate", #8
          "Cumulative GPA",    #9
          "Major Type",        #10
          "Grade in Intro Course", #11
          "Grade in Followup Course", #12
          "Grade in Fundamentals Course", #13
          "Grade in Systems Course", #14
          "Grade in Software Course", #15
          "Grade in Paradigms Course", #16
          "Instructor of Intro Course", #17
          "Instructor of Followup Course", #18
          "Instructor of Fundamentals Course", #19
          "Instructor of Systems Course" #20
          ]

# Read data from the csv file
def read_csv(READ_FILE):
    df = pd.read_csv(READ_FILE)
    #set headers for the columns
    df.columns = HEADERS
    return df

# Extra data from the dataset with only the selected attributes
# Drop the columns that have values missing
def data_with_attributes(dataset, attributes):
    dataset = pd.DataFrame(dataset, columns = attributes)
    dataset = dataset.dropna(axis=0)
    return dataset

# Split the dataset as per train_percentage into train and test dataset   
def split_dataset(feature_headers, feature_target, train_percentage):
    X_train, X_test, Y_train, Y_test = train_test_split(feature_headers, feature_target, test_size = train_percentage, random_state = 1, stratify=feature_target)

    # Print the sample sizes for our test and train data
    print ("X_train: ", X_train.shape)
    print ("X_test: ", X_test.shape)
    print ("Y_train: ", Y_train.shape)
    print ("Y_test: ", Y_test.shape)

    return X_train, X_test ,Y_train, Y_test

# Print the status of our training and testing sets
def print_sample_counts (Y, Y_train, Y_test):
    print ("Labels count in y: ", np.bincount(Y))
    print ("Labels count in y_train: ", np.bincount(Y_train))
    print ("Labels count in y_test: ", np.bincount(Y_test))

# Create decision tree for the given hypothesis
def create_decision_tree(X_train, X_test, Y_train, Y_test, depth, rand_state):
    tree = DecisionTreeClassifier(criterion='gini', max_depth=depth, random_state=rand_state, min_samples_leaf=20)
    tree.fit(X_train, Y_train)

    print ("Decision tree model score: ", tree.score(X_test, Y_test))
    return tree

# Feed data using random forest algorithm and then display which features in our selected
# attributes had significant impact during the classification
def random_forest_tree (X_train, X_test, Y_train, Y_test, dataset, hypothesis_feat_imp_img_file_name):
    # get the labels used for classification
    feat_labels = dataset.columns[:-1]
    # Generate random forest tree
    forest = RandomForestClassifier(criterion='gini', n_estimators=20, random_state=1, n_jobs=2, min_samples_leaf=20)
    forest.fit(X_train, Y_train)

    # Display the weightage of the features used during the classification using a bar graph
    importances = forest.feature_importances_
    indices = np.argsort(importances)[::-1]

    for f in range (X_train.shape[1]):
        print ("%2d) %-*s %f" % (f + 1, 30, 
                                feat_labels[indices[f]],
                                importances[indices[f]]))

    plt.title('Feature Importance')
    plt.bar(range(X_train.shape[1]),
            importances[indices],
            align='center')
    plt.xticks(range(X_train.shape[1]),
                feat_labels[indices], rotation=90)
    plt.xlim([-1, X_train.shape[1]])
    plt.tight_layout()
    plt.savefig(hypothesis_feat_imp_img_file_name, dpi=300)
    plt.show()

    # Feed testing data into our forest
    predictions = forest.predict(X_test)

    # Print the train and test accuracy of our model
    print ("Train Accuracy :: ", accuracy_score(Y_train, forest.predict(X_train)))
    print ("Test Accuracy :: ", accuracy_score(Y_test, predictions))
    return forest

# Export the decision tree that used DecisionTreeClassifier into a png file
def visualize_tree (tree, attributes, classes, output_png_filename):
    dot_data = export_graphviz(tree, 
                               filled=True,
                               rounded=True,
                               class_names=classes,
                               feature_names=attributes,
                               out_file=None)
    graph = graph_from_dot_data(dot_data)
    graph.write_png(output_png_filename)

################################################################################################
# Test our first hypothesis
def first_hypothesis(dataset, label_encoder):
    ########################################################################################
    # Hypothesis 1: Who has a high CGPA (greater than 3.0)?
    # Input parameters: Sex, First Generation, Race
    # Outcome Variable: Cumulative GPA > 3.0
    ########################################################################################
    attributes = ["Sex", "First Generation", "Race", "Cumulative GPA"]
    dataset = data_with_attributes(dataset, attributes)

    #Convert Sex variable to numeric
    encoded_sex = label_encoder.fit_transform(dataset["Sex"])

    # Pass data to split into test and training data
    # X = input parameters, Y = outcome variable
    X = pd.DataFrame ([encoded_sex, dataset["First Generation"], dataset["Race"]]).T
    Y = np.where(dataset["Cumulative GPA"] > str(3.0), 1, 0)
    X_train, X_test, Y_train, Y_test = split_dataset(X, Y, 0.3)

    # Print the sample sizes in our test and train samples
    print_sample_counts(Y, Y_train, Y_test)

    # create_decision_tree(X_train, X_test, Y_train, Y_test, maximum_depth, random_state)
    tree = create_decision_tree (X_train, X_test, Y_train, Y_test, 5, 1)

    # Print the decision tree using DecisionTreeClassifier
    feature_names= ["Sex", "First Generation", "Race"]    
    class_names= ["High CGPA", "Low CGPA"]
    visualize_tree (tree, feature_names, class_names, 'hypothesis1.png')    

    # Get accuracy score, feature importance bar graph using RandomForestClassifier
    tree = random_forest_tree (X_train, X_test, Y_train, Y_test, dataset, 'hyp1featureimp.png')
    print(tree)


# End of test of first hypothesis
###################################################################################################


###################################################################################################
# Test our second hypothesis
def second_hypothesis(dataset, label_encoder):
    ########################################################################################
    # Hypothesis 2: Do First Generation students graduate quickly in college?
    # Input parameters: Sex, First Generation, Cumulative GPA ( > 3.1)
    # Outcome Variable: Semesters taken to graduate (15)
    ########################################################################################
    attributes = ["Sex", "First Generation", "Cumulative GPA", "Semesters Taken To Graduate"]
    dataset = data_with_attributes(dataset, attributes)

    #Convert Sex variable to numeric
    encoded_sex = label_encoder.fit_transform(dataset["Sex"])
    dataset["Cumulative GPA"] = np.where(dataset["Cumulative GPA"] > str(3.1), 1, 0)

    # Pass data to split into test and training data
    # X = input parameters, Y = outcome variable
    X = pd.DataFrame ([encoded_sex,
                       dataset["First Generation"], 
                       dataset["Cumulative GPA"]]).T
    Y = np.where(dataset["Semesters Taken To Graduate"] <= str(15), 1, 0)
    X_train, X_test, Y_train, Y_test = split_dataset(X, Y, 0.28)

    # Print the sample sizes in our test and train samples
    print_sample_counts(Y, Y_train, Y_test)

    # create_decision_tree(X_train, X_test, Y_train, Y_test, maximum_depth, random_state)
    tree = create_decision_tree (X_train, X_test, Y_train, Y_test, 6, 1)

    # Print the decision tree using DecisionTreeClassifier
    feature_names= ["Sex", "First Generation", "Cumulative GPA"]  
    class_names = ["Semesters to Graduate > 15", "Semesters to Graduate <= 15"]  
    visualize_tree (tree, feature_names, class_names, 'hypothesis2.png')    

    # Get accuracy score, feature importance bar graph using RandomForestClassifier
    tree = random_forest_tree (X_train, X_test, Y_train, Y_test, dataset, 'hyp2featureimp.png')

# End of test of second hypothesis
###################################################################################################


###################################################################################################
# Test our third hypothesis
def third_hypothesis(dataset, label_encoder):
    ########################################################################################
    # Hypothesis 3: Who does well in Intro Course?
    # Input parameters: SAT Reading Score(>600), SAT Math Score(>715), High School GPA(>3.4), Instructor of Intro Course(>6)
    # Outcome Variable: Grade in Intro Course(>=3.5)
    ########################################################################################
    attributes = ["SAT Reading Score", "SAT Math Score", "High School GPA", "Instructor of Intro Course", "Grade in Intro Course"]
    dataset = data_with_attributes(dataset, attributes)

    dataset["SAT Reading Score"] = np.where(dataset["SAT Reading Score"] > 600, 1, 0)
    dataset["SAT Math Score"] = np.where(dataset["SAT Math Score"] > 715, 1, 0)
    dataset["High School GPA"] = np.where(dataset["High School GPA"] > 3.4, 1, 0)
    dataset["Instructor of Intro Course"] = np.where(dataset["Instructor of Intro Course"] > 6, 1, 0)
    
        
    # Pass data to split into test and training data
    # X = input parameters, Y = outcome variable
    X = pd.DataFrame ([dataset["SAT Reading Score"],
                       dataset["SAT Math Score"], 
                       dataset["High School GPA"],
                       dataset["Instructor of Intro Course"]]).T
    Y = np.where(dataset["Grade in Intro Course"] >= 3.5, 1, 0)
    X_train, X_test, Y_train, Y_test = split_dataset(X, Y, 0.28)

    # Print the sample sizes in our test and train samples
    print_sample_counts(Y, Y_train, Y_test)

    # create_decision_tree(X_train, X_test, Y_train, Y_test, maximum_depth, random_state)
    tree = create_decision_tree (X_train, X_test, Y_train, Y_test, 6, 1)

    # Print the decision tree using DecisionTreeClassifier
    feature_names= ["SAT Reading Score", "SAT Math Score", "High School GPA", "Instructor of Intro Course"]  
    class_names = ["< 3.5 in Intro Course", ">=3.5 in Intro Course"]  
    visualize_tree (tree, feature_names, class_names, 'hypothesis3.png')    

    # Get accuracy score, feature importance bar graph using RandomForestClassifier
    tree = random_forest_tree (X_train, X_test, Y_train, Y_test, dataset, 'hyp3featureimp.png')

# End of test of third hypothesis
###################################################################################################


###################################################################################################
# Test our fourth hypothesis
def fourth_hypothesis(dataset, label_encoder):
    ########################################################################################
    # Hypothesis 4: Who does well in Software Course?
    # Input parameters: Grade in Intro Course(>=3.5), Grade in Followup Course(>=3.0), Grade in Fundamentals Course(>=3.0)
    # Outcome Variable: Grade in Software Course(>=3.2)
    ########################################################################################
    attributes = ["Grade in Intro Course", "Grade in Followup Course", "Grade in Fundamentals Course", "Grade in Software Course"]
    dataset = data_with_attributes(dataset, attributes)

    dataset["Grade in Intro Course"] = np.where(dataset["Grade in Intro Course"] >= 3.5, 1, 0)
    dataset["Grade in Followup Course"] = np.where(dataset["Grade in Followup Course"] >= 3.0, 1, 0)
    dataset["Grade in Fundamentals Course"] = np.where(dataset["Grade in Fundamentals Course"] >= 3.0, 1, 0) 
        
    # Pass data to split into test and training data
    # X = input parameters, Y = outcome variable
    X = pd.DataFrame ([dataset["Grade in Intro Course"],
                       dataset["Grade in Followup Course"], 
                       dataset["Grade in Fundamentals Course"]]).T
    Y = np.where(dataset["Grade in Software Course"] >= 3.2, 1, 0)
    X_train, X_test, Y_train, Y_test = split_dataset(X, Y, 0.28)

    # Print the sample sizes in our test and train samples
    print_sample_counts(Y, Y_train, Y_test)

    # create_decision_tree(X_train, X_test, Y_train, Y_test, maximum_depth, random_state)
    tree = create_decision_tree (X_train, X_test, Y_train, Y_test, 6, 1)

    # Print the decision tree using DecisionTreeClassifier
    feature_names= ["Grade in Intro Course", "Grade in Followup Course", "Grade in Fundamentals Course"]  
    class_names = ["< 3.2 in Software Course", ">=3.2 in Software Course"]  
    visualize_tree (tree, feature_names, class_names, 'hypothesis4.png')    

    # Get accuracy score, feature importance bar graph using RandomForestClassifier
    tree = random_forest_tree (X_train, X_test, Y_train, Y_test, dataset, 'hyp4featureimp.png')

# End of test of fourth hypothesis
###################################################################################################


###################################################################################################
# Test our fifth hypothesis
def fifth_hypothesis(dataset, label_encoder):
    ########################################################################################
    # Hypothesis 5: How quickly do Sciences or Non-sciences students graduate?
    # Input parameters: Sex, Major Type, Cumulative GPA (>3.2)
    # Outcome Variable: Semesters Taken to Graduate (<=15)
    ########################################################################################
    attributes = ["Sex", "Major Type", "Cumulative GPA", "Semesters Taken To Graduate"]
    dataset = data_with_attributes(dataset, attributes)

    #Convert variables to numeric
    encoded_sex = label_encoder.fit_transform(dataset["Sex"])
    # Separate into Sciences (CS, Math, Sciences) vs Non-Science
    encoded_majors = np.where(dataset["Major Type"] > 3, 1, 0)

    dataset["Cumulative GPA"] = np.where(dataset["Cumulative GPA"] > str(3.2), 1, 0)

    # Pass data to split into test and training data
    # X = input parameters, Y = outcome variable
    X = pd.DataFrame ([encoded_sex,
                       encoded_majors, 
                       dataset["Cumulative GPA"]]).T
    Y = np.where(dataset["Semesters Taken To Graduate"] <= str(15), 1, 0)
    X_train, X_test, Y_train, Y_test = split_dataset(X, Y, 0.28)

    # Print the sample sizes in our test and train samples
    print_sample_counts(Y, Y_train, Y_test)

    # create_decision_tree(X_train, X_test, Y_train, Y_test, maximum_depth, random_state)
    tree = create_decision_tree (X_train, X_test, Y_train, Y_test, 6, 1)

    # Print the decision tree using DecisionTreeClassifier
    feature_names= ["Sex", "Major Type", "Cumulative GPA"]  
    class_names = ["Semesters to Graduate > 15", "Semesters to Graduate <= 15"]
    visualize_tree (tree, feature_names, class_names, 'hypothesis5.png')    

    # Get accuracy score, feature importance bar graph using RandomForestClassifier
    tree = random_forest_tree (X_train, X_test, Y_train, Y_test, dataset, 'hyp5featureimp.png')

# End of test of fifth hypothesis
###################################################################################################


# Entry point to our program
def main():
    # Load the csv file into pandas dataframe
    dataset = read_csv(READ_FILE)
    label_encoder = preprocessing.LabelEncoder()

    print ("First Hypothesis: ")
    first_hypothesis(dataset, label_encoder)

    print (" ")
    print ("Second Hypothesis: ")
    second_hypothesis(dataset, label_encoder)

    print (" ")
    print ("Third Hypothesis: ")
    third_hypothesis(dataset, label_encoder)
    
    print (" ")
    print ("Fourth Hypothesis: ")
    fourth_hypothesis(dataset, label_encoder)
    
    print (" ")
    print ("Fifth Hypothesis: ")
    fifth_hypothesis(dataset, label_encoder)

if __name__ == "__main__":
    main()