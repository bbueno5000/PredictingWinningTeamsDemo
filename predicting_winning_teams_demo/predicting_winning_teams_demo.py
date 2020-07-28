"""
DOCSTRING
"""
import datetime
import itertools
import pandas
import sklearn
import time 
import xgboost

class Cleaning:
    """
    DOCSTRING
    """
    def __init__(self):
        raw_data = list()
        raw_data.append(pandas.read_csv('datasets/2000-01.csv'))
        raw_data.append(pandas.read_csv('datasets/2001-02.csv'))
        raw_data.append(pandas.read_csv('datasets/2002-03.csv'))
        raw_data.append(pandas.read_csv('datasets/2003-04.csv'))
        raw_data.append(pandas.read_csv('datasets/2004-05.csv'))
        raw_data.append(pandas.read_csv('datasets/2005-06.csv'))
        raw_data.append(pandas.read_csv('datasets/2006-07.csv'))
        raw_data.append(pandas.read_csv('datasets/2007-08.csv'))
        raw_data.append(pandas.read_csv('datasets/2008-09.csv'))
        raw_data.append(pandas.read_csv('datasets/2009-10.csv'))
        raw_data.append(pandas.read_csv('datasets/2010-11.csv'))
        raw_data.append(pandas.read_csv('datasets/2011-12.csv'))
        raw_data.append(pandas.read_csv('datasets/2012-13.csv'))
        raw_data.append(pandas.read_csv('datasets/2013-14.csv'))
        raw_data.append(pandas.read_csv('datasets/2014-15.csv'))
        raw_data.append(pandas.read_csv('datasets/2015-16.csv'))

        def parse_date(date):
            if date == '':
                return None
            return datetime.datetime.strptime(date, '%d/%m/%y').date()
    
        def parse_date_other(date):
            if date == '':
                return None
            return datetime.datetime.strptime(date, '%d/%m/%Y').date()

        for i in range(len(raw_data)):
            if i == 2:
                raw_data[i].Date = raw_data[i].Date.apply(parse_date_other)
            else:
                raw_data[i].Date = raw_data[i].Date.apply(parse_date)

        columns_req = ['Date', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'FTR']

        playing_statistics = list()
        for data in raw_data:
            playing_statistics.append(data[columns_req])

        playing_statistics = pandas.DataFrame(playing_statistics)
        playing_statistics = self.get_gss(playing_statistics)
        playing_statistics = self.get_agg_points(playing_statistics)
        playing_statistics = self.add_form_df(playing_statistics)

        cols = [
            'Date', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'FTR', 'HTGS',
            'ATGS', 'HTGC', 'ATGC', 'HTP', 'ATP', 'HM1', 'HM2', 'HM3',
            'HM4', 'HM5', 'AM1', 'AM2', 'AM3', 'AM4', 'AM5']

        for stats in playing_statistics:
            stats = stats[cols]

        #standings = pandas.read_csv(loc + "EPLStandings.csv")
        #standings.set_index(['Team'], inplace=True)
        #standings = standings.fillna(18)

        playing_statistics = self.get_mw(playing_statistics)

        playing_stats = pandas.concat(
            [playing_statistics[0],
             playing_statistics[1],
             playing_statistics[2],
             playing_statistics[3],
             playing_statistics[4],
             playing_statistics[5],
             playing_statistics[6],
             playing_statistics[7],
             playing_statistics[8],
             playing_statistics[9],
             playing_statistics[10],
             playing_statistics[11],
             playing_statistics[12],
             playing_statistics[13],
             playing_statistics[14],
             playing_statistics[15]],
             ignore_index=True)

        playing_stats['HTFormPtsStr'] = playing_stats['HM1'] + playing_stats['HM2'] \
            + playing_stats['HM3'] + playing_stats['HM4'] + playing_stats['HM5']
        playing_stats['ATFormPtsStr'] = playing_stats['AM1'] + playing_stats['AM2'] \
            + playing_stats['AM3'] + playing_stats['AM4'] + playing_stats['AM5']
        playing_stats['HTFormPts'] = playing_stats['HTFormPtsStr'].apply(self.get_form_points)
        playing_stats['ATFormPts'] = playing_stats['ATFormPtsStr'].apply(self.get_form_points)
        playing_stats['HTWinStreak3'] = playing_stats['HTFormPtsStr'].apply(self.get_3game_ws)
        playing_stats['HTWinStreak5'] = playing_stats['HTFormPtsStr'].apply(self.get_5game_ws)
        playing_stats['HTLossStreak3'] = playing_stats['HTFormPtsStr'].apply(self.get_3game_ls)
        playing_stats['HTLossStreak5'] = playing_stats['HTFormPtsStr'].apply(self.get_5game_ls)
        playing_stats['ATWinStreak3'] = playing_stats['ATFormPtsStr'].apply(self.get_3game_ws)
        playing_stats['ATWinStreak5'] = playing_stats['ATFormPtsStr'].apply(self.get_5game_ws)
        playing_stats['ATLossStreak3'] = playing_stats['ATFormPtsStr'].apply(self.get_3game_ls)
        playing_stats['ATLossStreak5'] = playing_stats['ATFormPtsStr'].apply(self.get_5game_ls)
        playing_stats.keys()
        # goal difference
        playing_stats['HTGD'] = playing_stats['HTGS'] - playing_stats['HTGC']
        playing_stats['ATGD'] = playing_stats['ATGS'] - playing_stats['ATGC']
        # difference in points
        playing_stats['DiffPts'] = playing_stats['HTP'] - playing_stats['ATP']
        playing_stats['DiffFormPts'] = playing_stats['HTFormPts'] - playing_stats['ATFormPts']
        # difference in last year positions
        playing_stats['DiffLP'] = playing_stats['HomeTeamLP'] - playing_stats['AwayTeamLP']
        # scale difference points, difference form points, HTGD, ATGD by Matchweek
        cols = ['HTGD','ATGD','DiffPts','DiffFormPts','HTP','ATP']
        playing_stats.MW = playing_stats.MW.astype(float)
        for col in cols:
            playing_stats[col] = playing_stats[col] / playing_stats.MW
        playing_stats['FTR'] = playing_stats.FTR.apply(self.only_hw)
        # testing set (2015-16 season)
        playing_stat_test = playing_stats[5700:]
        playing_stats.to_csv(loc + "final_dataset.csv")
        playing_stat_test.to_csv(loc+"test.csv")

    def add_form(self, playing_stats, num):
        """
        DOCSTRING
        """
        form = self.get_form(playing_stats, num)
        h = ['M' for i in range(num * 10)] # since form is not available for n MW (n*10)
        a = ['M' for i in range(num * 10)]
        j = num
        for i in range((num * 10), 380):
            ht = playing_stats.iloc[i].HomeTeam
            at = playing_stats.iloc[i].AwayTeam
            past = form.loc[ht][j] # get past n results
            h.append(past[num-1]) # 0 index is most recent
            past = form.loc[at][j] # get past n results
            a.append(past[num-1]) # 0 index is most recent
            if ((i + 1) % 10) == 0:
                j += 1
        playing_stats['HM' + str(num)] = h
        playing_stats['AM' + str(num)] = a
        return playing_stats

    def add_form_df(self, playing_statistics):
        """
        DOCSTRING
        """
        playing_statistics = self.add_form(playing_statistics, 1)
        playing_statistics = self.add_form(playing_statistics, 2)
        playing_statistics = self.add_form(playing_statistics, 3)
        playing_statistics = self.add_form(playing_statistics, 4)
        playing_statistics = self.add_form(playing_statistics, 5)
        return playing_statistics

    def get_3game_ls(self, string):
        """
        DOCSTRING
        """
        if string[-3:] == 'LLL':
            return 1
        return 0

    def get_3game_ws(self, string):
        """
        Identify Win/Loss streaks if any.
        """
        if string[-3:] == 'WWW':
            return 1
        return 0
        
    def get_5game_ls(self, string):
        """
        DOCSTRING
        """
        if string == 'LLLLL':
            return 1
        return 0

    def get_5game_ws(self, string):
        """
        Identify Win/Loss streaks if any.
        """
        if string == 'WWWWW':
            return 1
        return 0

    def get_agg_points(self, playing_stats):
        """
        DOCSTRING
        """
        matches = self.get_matches(playing_stats)
        cum_pts = self.get_cuml_points(matches)
        ATP, HTP, j = [], [], 0
        for i in range(380):
            ht = playing_stats.iloc[i].HomeTeam
            at = playing_stats.iloc[i].AwayTeam
            HTP.append(cum_pts.loc[ht][j])
            ATP.append(cum_pts.loc[at][j])
            if ((i + 1) % 10) == 0:
                j += 1
        playing_stats['HTP'] = HTP
        playing_stats['ATP'] = ATP
        return playing_stats

    def get_cuml_points(self, matches):
        """
        DOCSTRING
        """
        matches_points = matches.applymap(self.get_points)
        for i in range(2, 39):
            matchres_points[i] = matches_points[i] + matches_points[i-1]
        matches_points.insert(column=0, loc=0, value=[0*i for i in range(20)])
        return matches_points

    def get_form(self, playing_stats, num):
        """
        DOCSTRING
        """
        form = self.get_matches(playing_stats)
        form_final = form.copy()
        for i in range(num,39):
            form_final[i] = ''
            j = 0
            while j < num:
                form_final[i] += form[i-j]
                j += 1
        return form_final

    def get_form_points(self, string):
        """
        Gets the form points.
        """
        sum = 0
        for letter in string:
            sum += self.get_points(letter)
        return sum

    def get_goals_conceded(self, playing_stats):
        """
        Gets the goals conceded, arranged by teams and match week.
        The value corresponding to keys is a list containing the match location.
        """
        teams = {}
        for i in playing_stats.groupby('HomeTeam').mean().T.columns:
            teams[i] = []
        for i in range(len(playing_stats)):
            ATGC = playing_stats.iloc[i]['FTHG']
            HTGC = playing_stats.iloc[i]['FTAG']
            teams[playing_stats.iloc[i].HomeTeam].append(HTGC)
            teams[playing_stats.iloc[i].AwayTeam].append(ATGC)
        GoalsConceded = pandas.DataFrame(data=teams, index = [i for i in range(1, 39)]).T
        GoalsConceded[0] = 0
        for i in range(2,39):
            GoalsConceded[i] = GoalsConceded[i] + GoalsConceded[i-1]
        return GoalsConceded

    def get_goals_scored(self, playing_stats):
        """
        Gets the goals scored, arranged by teams and match week.
        The value corresponding to keys is a list containing the match location.
        """
        teams = {}
        for i in playing_stats.groupby('HomeTeam').mean().T.columns:
            teams[i] = []
        for i in range(len(playing_stats)):
            HTGS = playing_stats.iloc[i]['FTHG']
            ATGS = playing_stats.iloc[i]['FTAG']
            teams[playing_stats.iloc[i].HomeTeam].append(HTGS)
            teams[playing_stats.iloc[i].AwayTeam].append(ATGS)
        GoalsScored = pandas.DataFrame(data=teams, index = [i for i in range(1, 39)]).T
        GoalsScored[0] = 0
        for i in range(2, 39):
            GoalsScored[i] = GoalsScored[i] + GoalsScored[i-1]
        return GoalsScored

    def get_gss(self, playing_stats):
        """
        DOCSTRING
        """
        GC = self.get_goals_conceded(playing_stats)
        GS = self.get_goals_scored(playing_stats)
        ATGC, ATGS, HTGC, HTGS, j = [], [], [], [], 0
        for i in range(380):
            ht = playing_stats.iloc[i].HomeTeam
            at = playing_stats.iloc[i].AwayTeam
            HTGS.append(GS.loc[ht][j])
            ATGS.append(GS.loc[at][j])
            HTGC.append(GC.loc[ht][j])
            ATGC.append(GC.loc[at][j])
            if ((i + 1) % 10) == 0:
                j += 1
        playing_stats['HTGS'] = HTGS
        playing_stats['ATGS'] = ATGS
        playing_stats['HTGC'] = HTGC
        playing_stats['ATGC'] = ATGC
        return playing_stats

    def get_last(self, playing_stats, standings, year):
        """
        DOCSTRING
        """
        AwayTeamLP, HomeTeamLP = [], []
        for i in range(380):
            ht = playing_stats.iloc[i].HomeTeam
            at = playing_stats.iloc[i].AwayTeam
            HomeTeamLP.append(standings.loc[ht][year])
            AwayTeamLP.append(standings.loc[at][year])
        playing_stats['HomeTeamLP'] = HomeTeamLP
        playing_stats['AwayTeamLP'] = AwayTeamLP
        return playing_stats
        
    def get_matches(self, playing_stats):
        """
        Create a dictionary with team names as keys.
        The value corresponding to keys is a list containing the match result.
        """
        teams = {}
        for i in playing_stats.groupby('HomeTeam').mean().T.columns:
            teams[i] = []
        for i in range(len(playing_stats)):
            if playing_stats.iloc[i].FTR == 'H':
                teams[playing_stats.iloc[i].HomeTeam].append('W')
                teams[playing_stats.iloc[i].AwayTeam].append('L')
            elif playing_stats.iloc[i].FTR == 'A':
                teams[playing_stats.iloc[i].AwayTeam].append('W')
                teams[playing_stats.iloc[i].HomeTeam].append('L')
            else:
                teams[playing_stats.iloc[i].AwayTeam].append('D')
                teams[playing_stats.iloc[i].HomeTeam].append('D')
        return pandas.DataFrame(data=teams, index = [i for i in range(1, 39)]).T

    def get_mw(self, playing_stats):
        """
        DOCSTRING
        """
        match_week, j = [], 1
        for i in range(380):
            match_week.append(j)
            if ((i + 1) % 10) == 0:
                j += 1
        playing_stats['MW'] = match_week
        return playing_stats

    def get_points(self, result):
        """
        DOCSTRING
        """
        if result == 'W':
            return 3
        elif result == 'D':
            return 1
        return 0

    def only_hw(self, string):
        """
        DOCSTRING
        """
        if string == 'H':
            return 'H'
        return 'NH'

class Prediction:
    """
    DOCSTRING
    """
    def __init__(self):
        data = pandas.read_csv('final_dataset.csv')
        print(data.head())
        # Full Time Result (H=Home Win, D=Draw, A=Away Win)
        # HTGD - Home team goal difference
        # ATGD - away team goal difference
        # HTP - Home team points
        # ATP - Away team points
        # DiffFormPts Diff in points
        # DiffLP - Differnece in last years prediction
        # Input - 12 other features (fouls, shots, goals, misses,corners, red card, yellow cards)
        # Output - Full Time Result (H=Home Win, D=Draw, A=Away Win)
        n_matches = data.shape[0]
        n_features = data.shape[1] - 1
        n_homewins = len(data[data.FTR == 'H'])
        win_rate = (float(n_homewins) / (n_matches)) * 100
        print("Total number of matches: {}".format(n_matches))
        print("Number of features: {}".format(n_features))
        print("Number of matches won by home team: {}".format(n_homewins))
        print("Win rate of home team: {:.2f}%".format(win_rate))

    def __call__(self):
        self.visualize_distribution()
        self.standardize_data()
        self.train_classifier()
        self.predict_labels()
        self.train_predict()
        self.initialize_models()
        self.tune_xgb_classifier()
            
    def initialize_models(self):
        """
        DOCSTRING
        """
        clf_A = sklearn.linear_model.LogisticRegression(random_state = 42)
        clf_B = sklearn.svm.SVC(random_state = 912, kernel='rbf')
        # boosting refers to this general problem of producing a very accurate prediction rule,
        # by combining rough and moderately inaccurate rules-of-thumb
        clf_C = xgboost.XGBClassifier(seed = 82)
        train_predict(clf_A, X_train, y_train, X_test, y_test)
        print('')
        train_predict(clf_B, X_train, y_train, X_test, y_test)
        print('')
        train_predict(clf_C, X_train, y_train, X_test, y_test)
        print('')

    def predict_labels(clf, features, target):
        """
        Makes predictions using a fit classifier based on F1 score.
        """
        start = time.time()
        y_pred = clf.predict(features)
        end = time.time()
        print("Made predictions in {:.4f} seconds.".format(end - start))
        return sklearn.metrics.f1_score(target, y_pred, pos_label='H'), sum(target == y_pred) / float(len(y_pred))

    def preprocess_features(X):
        """
        Preprocesses the football data and converts catagorical variables into dummy variables.
        We want continous vars that are integers for our input data,
        so lets remove any categorical variables.
        """
        output = pandas.DataFrame(index = X.index)
        for col, col_data in X.iteritems():
            if col_data.dtype == object:
                col_data = pandas.get_dummies(col_data, prefix = col)
            output = output.join(col_data)
        return output

    def standardize_data(self):
        """
        DOCSTRING
        """
        # Separate into feature set and target variable
        # FTR = Full Time Result (H=Home Win, D=Draw, A=Away Win)
        X_all = data.drop(['FTR'], 1)
        y_all = data['FTR']
        # center to the mean and component wise scale to unit variance
        cols = [['HTGD','ATGD','HTP','ATP','DiffLP']]
        for col in cols:
            X_all[col] = sklearn.preprocessing.scale(X_all[col])
        # last 3 wins for both sides
        X_all.HM1 = X_all.HM1.astype('str')
        X_all.HM2 = X_all.HM2.astype('str')
        X_all.HM3 = X_all.HM3.astype('str')
        X_all.AM1 = X_all.AM1.astype('str')
        X_all.AM2 = X_all.AM2.astype('str')
        X_all.AM3 = X_all.AM3.astype('str')
        X_all = preprocess_features(X_all)
        print("Processed feature columns ({} total features):{}".format(
            len(X_all.columns), list(X_all.columns)))
        print("Feature values:")
        print(X_all.head())
        X_train, X_test, y_train, y_test = sklearn.cross_validation.train_test_split(
            X_all, y_all, test_size = 50, random_state = 2, stratify = y_all)
    
    def train_classifier(clf, X_train, y_train):
        """
        Fits a classifier to the training data.
        """
        start = time.time()
        clf.fit(X_train, y_train)
        end = time.time()
        print("Trained model in {:.4f} seconds".format(end - start))
    
    def train_predict(clf, X_train, y_train, X_test, y_test):
        """
        Train and predict using a classifer based on F1 score.
        """
        # Indicate the classifier and the training set size
        print("Training a {} using a training set size of {}. . .".format(
            clf.__class__.__name__, len(X_train)))
        # Train the classifier
        train_classifier(clf, X_train, y_train)
        # Print the results of prediction for both training and testing
        f1, acc = predict_labels(clf, X_train, y_train)
        print(f1, acc)
        print("F1 score and accuracy score for training set: {:.4f} , {:.4f}.".format(f1 , acc))
        f1, acc = predict_labels(clf, X_test, y_test)
        print("F1 score and accuracy score for test set: {:.4f} , {:.4f}.".format(f1 , acc))
    
    def tune_xgb_classifier(self):
        """
        DOCSTRING
        """
        parameters = {
            'learning_rate': [0.1],
            'n_estimators': [40],
            'max_depth': [3],
            'min_child_weight': [3],
            'gamma': [0.4],
            'subsample': [0.8],
            'colsample_bytree': [0.8],
            'scale_pos_weight': [1],
            'reg_alpha': [1e-5]}  
        clf = xgboost.XGBClassifier(seed=2)
        f1_scorer = sklearn.metrics.make_scorer(sklearn.metrics.f1_score,pos_label='H')
        grid_obj = sklearn.grid_search.GridSearchCV(clf, scoring=f1_scorer, param_grid=parameters, cv=5)
        grid_obj = grid_obj.fit(X_train,y_train)
        clf = grid_obj.best_estimator_
        print(clf)
        f1, acc = predict_labels(clf, X_train, y_train)
        print("F1 score and accuracy score for training set: {:.4f} , {:.4f}.".format(f1 , acc))
        f1, acc = predict_labels(clf, X_test, y_test)
        print("F1 score and accuracy score for test set: {:.4f} , {:.4f}.".format(f1 , acc))

    def visualize_distribution(self):
        """
        HTGD - Home team goal difference
        ATGD - away team goal difference
        HTP - Home team points
        ATP - Away team points
        DiffFormPts Diff in points
        DiffLP - Differnece in last years prediction
        """
        pandas.tools.plotting.scatter_matrix(
            data[['HTGD', 'ATGD', 'HTP', 'ATP', 'DiffFormPts', 'DiffLP']],
            figsize=(10,10))

if __name__ == '__main__':
    cleaning = Cleaning()
    cleaning()
