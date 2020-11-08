import abc
import collections
import datetime as dt
import functools
import inspect
import pathlib
import pickle
import sys
import time

import chime
import pandas as pd
from scipy import stats
import tqdm


data_dir = pathlib.Path('data')

if data_dir.joinpath('train.pkl').exists():
    print('Loading .pkl')
    train = pd.read_pickle(data_dir.joinpath('train.pkl'))

else:

    print('Loading .csv')
    dtypes = {
        'row_id': 'int64',
        'timestamp': 'int64',
        'user_id': 'int32',
        'content_id': 'int16',
        'content_type_id': 'int8',
        'task_container_id': 'int16',
        'user_answer': 'int8',
        'answered_correctly': 'int8',
        'prior_question_elapsed_time': 'float32',
        'prior_question_had_explanation': 'boolean'
    }
    train = pd.read_csv(
        data_dir.joinpath('train.csv'),
        index_col='row_id',
        dtype=dtypes
    )

    # The `task_container_id` variable is supposed to be monotonically increasing for each user.
    # But that doesn't seem to be the case. For instance, see user 115.
    # Therefore, I renumber the tasks to make sure they're monotonically increasing for each user.
    train['task_container_id'] = train.groupby('user_id')['task_container_id'].transform(lambda x: pd.factorize(x)[0]).astype('int16')

    train.to_pickle(data_dir.joinpath('train.pkl'))

#print(train.head(5))


# Parts

question_parts = (
    pd.read_csv('data/questions.csv', usecols=['question_id', 'part'])
    .rename(columns={'question_id': 'content_id'})
    .assign(content_type_id=0)
)

lecture_parts = (
    pd.read_csv('data/lectures.csv', usecols=['lecture_id', 'part'])
    .rename(columns={'lecture_id': 'content_id'})
    .assign(content_type_id=1)
)

parts = pd.concat((question_parts, lecture_parts))
parts = parts.set_index(['content_type_id', 'content_id'])['part']
parts = parts.map({
    1: 'photographs',
    2: 'question_response',
    3: 'conversations',
    4: 'talks',
    5: 'incomplete_sentences',
    6: 'text_completion',
    7: 'passages'
})
parts = parts.astype('category')
parts.to_pickle('features/parts.pkl')

# We can now iterate over batches of the training data. The idea is that each batch is going to
# behave like the data that the `env.iter_test` function will yield in the Kaggle kernel. We will
# thus call each batch a "group" to adopt the same terminology.

def iter_groups(train):

    prev_group = pd.DataFrame()

    for _, group in iter(train.groupby('task_container_id')):
        group = group.join(parts, on=['content_type_id', 'content_id'])
        questions = group.query('content_type_id == 0')
        yield questions, prev_group
        prev_group = group

#groups = iter_groups(train[:10_000])
#questions, prev_group = next(groups)
#print(questions.head())
#print(prev_group.head())
#next_questions, next_prev_group = next(groups)
#print(next_questions.head())
#print(next_prev_group.head())

# As you can see, the first group contains the first interaction of each user. The next group
# contains the second interaction, along with the correctness information for the first group.

# The goal is now to build stateful feature extractors. Each such feature extractor should provide
# the ability to produce features for each row in a group. The feature extractor should then be
# able to update itself with the new information provided by the group. Here is the interface:

class Extractor(abc.ABC):

    def __str__(self):
        return self.__class__.__name__

    @abc.abstractmethod
    def transform(self, questions):
        pass


class StatefulExtractor(Extractor):

    update_during_train = True

    @abc.abstractmethod
    def update(self, questions, prev_group):
        pass


class AvgCorrect(StatefulExtractor):

    def __init__(self, prior_mean, prior_size):
        self.prior_mean = prior_mean
        self.prior_size = prior_size
        self.stats = pd.DataFrame(columns=['mean', 'size'], dtype=float)

    def __str__(self):
        return f'{self.__class__.__name__}_prior_mean={self.prior_mean}_prior_size={self.prior_size}'

    def update(self, questions, prev_group):

        # Initialize statistics for new users
        new = pd.Index(questions['user_id']).difference(self.stats.index)
        if len(new) > 0:
            prior = pd.DataFrame(
                {'mean': self.prior_mean, 'size': self.prior_size},
                index=new
            )
            self.stats = self.stats.append(prior)

        # Nothing to do if nothing happened before
        if len(prev_group) == 0:
            return

        # Compute the new statistics
        stats = (
            prev_group
            .query('content_type_id == 0')
            .groupby('user_id')['answered_correctly']
            .agg(['mean', 'size'])
        )

        # Update the old statistics with the new statistics
        users = stats.index
        m = stats.loc[users, 'size']
        self.stats.loc[users, 'size'] += m
        n = self.stats.loc[users, 'size']
        avg = self.stats.loc[users, 'mean']
        new_avg = stats.loc[users, 'mean']
        self.stats.loc[users, 'mean'] += m * (new_avg - avg) / n

    def transform(self, questions):
        avgs = self.stats.loc[questions['user_id'], 'mean'].rename('avg_correct')
        avgs.index = questions.index
        return avgs


class QuestionDifficulty(StatefulExtractor):

    update_during_train = False

    def __init__(self, train):
        self.stats = (
            train
            .query('content_type_id == 0')
            .groupby('content_id')['answered_correctly']
            .agg(['mean', 'size'])
        )

    def update(self, questions, prev_group):

        if len(prev_group) == 0:
            return

        # Compute the new statistics
        stats = (
            prev_group
            .query('content_type_id == 0')
            .groupby('content_id')['answered_correctly']
            .agg(['mean', 'size'])
        )

        # Update the old statistics with the new statistics
        question_ids = stats.index
        m = stats.loc[question_ids, 'size']
        self.stats.loc[question_ids, 'size'] += m
        n = self.stats.loc[question_ids, 'size']
        avg = self.stats.loc[question_ids, 'mean']
        new_avg = stats.loc[question_ids, 'mean']
        self.stats.loc[question_ids, 'mean'] += m * (new_avg - avg) / n

    def transform(self, questions):
        avgs = self.stats.loc[questions['content_id'], 'mean'].rename('question_difficulty')
        avgs.index = questions.index
        return avgs


class Part(Extractor):

    def transform(self, questions):
        return questions['part']


class Timestamp(Extractor):

    def transform(self, questions):
        return questions['timestamp']


class BundleSize(Extractor):

    def transform(self, questions):
        bundle_sizes = questions['user_id'].value_counts().rename('bundle_size')
        return questions.join(bundle_sizes, on='user_id')['bundle_size']


class BundlePosition(Extractor):

    def transform(self, questions):
        return questions.groupby('user_id').cumcount().rename('bundle_position')


class UserQuestionCount(StatefulExtractor):

    def __init__(self):
        self.counts = pd.Series(dtype='uint16', index=pd.Index([], name='user_id'))

    def update(self, questions, prev_group):

        # Initialize statistics for new users
        new = pd.Index(questions['user_id']).difference(self.counts.index)
        if len(new) > 0:
            self.counts = self.counts.append(pd.Series(0, index=new))

        # Nothing to do if nothing happened before
        if len(prev_group) == 0:
            return

        new_counts = (
            prev_group
            .query('content_type_id == 0')
            .groupby('user_id')
            .size()
            .astype('uint16')
        )

        self.counts[new_counts.index] += new_counts

    def transform(self, questions):
        counts = self.counts[questions['user_id']]
        counts.index = questions.index
        counts = counts.rename('user_question_count')
        return counts


class UserLectureCount(StatefulExtractor):

    def __init__(self):
        self.counts = pd.Series(dtype='uint16', index=pd.Index([], name='user_id'))

    def update(self, questions, prev_group):

        # Initialize statistics for new users
        new = pd.Index(questions['user_id']).difference(self.counts.index)
        if len(new) > 0:
            self.counts = self.counts.append(pd.Series(0, index=new))

        # Nothing to do if nothing happened before
        if len(prev_group) == 0:
            return

        new_counts = (
            prev_group
            .query('content_type_id == 1')
            .groupby('user_id')
            .size()
            .astype('uint16')
        )

        self.counts[new_counts.index] += new_counts

    def transform(self, questions):
        counts = self.counts[questions['user_id']]
        counts.index = questions.index
        counts = counts.rename('user_lecture_count')
        return counts


class UserQuestionAvgDuration(StatefulExtractor):

    def __init__(self):
        self.stats = pd.DataFrame(columns=['mean', 'size'], dtype=float)

    def update(self, questions, prev_group):

        # Initialize statistics for new users
        new = pd.Index(questions['user_id']).difference(self.stats.index)
        if len(new) > 0:
            init_stats = pd.DataFrame({'mean': 0, 'size': 0}, index=new)
            self.stats = self.stats.append(init_stats)

        if len(prev_group) == 0:
            return

        prev_times = (
            questions[questions['prior_question_elapsed_time'].notnull()]
            .rename(columns={'prior_question_elapsed_time': 'elapsed_time'})
            [['user_id', 'elapsed_time']]
            .groupby('user_id')
            .first()['elapsed_time']
        )

        stats = (
            prev_group
            .query('content_type_id == 0')
            .join(prev_times, on='user_id', how='inner')
            .groupby('user_id')['elapsed_time']
            .agg(['mean', 'size'])
        )

        # Update the old statistics with the new statistics
        users = stats.index
        m = stats.loc[users, 'size']
        self.stats.loc[users, 'size'] += m
        n = self.stats.loc[users, 'size']
        avg = self.stats.loc[users, 'mean']
        new_avg = stats.loc[users, 'mean']
        self.stats.loc[users, 'mean'] += m * (new_avg - avg) / n

    def transform(self, questions):
        avgs = self.stats.loc[questions['user_id'], 'mean']
        avgs.index = questions.index
        avgs = avgs.rename('user_question_avg_duration')
        return avgs


class UserExpAvgCorrect(StatefulExtractor):

    def __init__(self, prior_mean, alpha):
        self.prior_mean = prior_mean
        self.alpha = alpha
        self.stats = pd.Series(dtype=float)

    def __str__(self):
        return f'{self.__class__.__name__}_prior_mean={self.prior_mean}_alpha={self.alpha}'

    def update(self, questions, prev_group):

        # Initialize statistics for new users
        new = pd.Index(questions['user_id']).difference(self.stats.index)
        if len(new) > 0:
            init_stats = pd.Series(self.prior_mean, index=new)
            self.stats = self.stats.append(init_stats)

        # Nothing to do if nothing happened before
        if len(prev_group) == 0:
            return

        new_stats = (
            prev_group
            .query('content_type_id == 0')
            .groupby('user_id')['answered_correctly']
            .agg('mean')
        )

        old_stats = self.stats[new_stats.index]

        self.stats[new_stats.index] = self.alpha * new_stats + (1 - self.alpha) * old_stats

    def transform(self, questions):
        avgs = self.stats[questions['user_id']]
        avgs.index = questions.index
        avgs = avgs.rename('user_expo_avg_correct')
        return avgs


class DejaVu(StatefulExtractor):

    def __init__(self):
        self.correct = collections.defaultdict(functools.partial(collections.defaultdict, int))
        self.incorrect = collections.defaultdict(functools.partial(collections.defaultdict, int))

    def update(self, questions, prev_group):

        if len(prev_group) == 0:
            return

        for r in prev_group.itertuples():
            if r.answered_correctly == 1:
                self.correct[r.content_id][r.user_id] += 1
            elif r.answered_correctly == 0:
                self.incorrect[r.content_id][r.user_id] += 1

    def transform(self, questions):
        deja = pd.DataFrame({
            'deja_vu_correct': [self.correct[r.content_id][r.user_id] for r in questions.itertuples()],
            'deja_vu_incorrect': [self.incorrect[r.content_id][r.user_id] for r in questions.itertuples()],
        })
        deja.index = questions.index
        return deja


class QuestionNChoices(Extractor):

    def __init__(self, train):
        self.n_choices = (
            train
            .query('content_type_id == 0')
            .groupby('content_id')['user_answer'].max()
            .astype('uint8')
        ).rename('question_n_choices')
        self.n_choices += 1

    def transform(self, questions):
        n_choices = self.n_choices.loc[questions['content_id']]
        n_choices.index = questions.index
        return n_choices


class QuestionAnswerEntropy(Extractor):

    def __init__(self, train):
        question_answer_dist = train.query('content_type_id == 0').groupby(['content_id', 'user_answer']).size()
        self.question_answer_entropy = question_answer_dist.groupby('content_id').apply(lambda counts: stats.entropy(counts + 50))
        self.question_answer_entropy = self.question_answer_entropy.rename('question_answer_entropy')

    def transform(self, questions):
        question_answer_entropy = self.question_answer_entropy.loc[questions['content_id']]
        question_answer_entropy.index = questions.index
        return question_answer_entropy


class UserPartCount(StatefulExtractor):

    def __init__(self):
        self.correct = collections.defaultdict(functools.partial(collections.defaultdict, int))
        self.incorrect = collections.defaultdict(functools.partial(collections.defaultdict, int))

    def update(self, questions, prev_group):

        if len(prev_group) == 0:
            return

        for r in prev_group.itertuples():
            if r.answered_correctly == 1:
                self.correct[r.part][r.user_id] += 1
            elif r.answered_correctly == 0:
                self.incorrect[r.part][r.user_id] += 1

    def transform(self, questions):
        deja = pd.DataFrame({
            'part_correct': [self.correct[r.part][r.user_id] for r in questions.itertuples()],
            'part_incorrect': [self.incorrect[r.part][r.user_id] for r in questions.itertuples()],
        })
        deja.index = questions.index
        return deja


# Extracting features for the training set

extractors = [
    AvgCorrect(.6, 20),
    QuestionDifficulty(train),
    Part(),
    BundleSize(),
    BundlePosition(),
    UserQuestionCount(),
    UserLectureCount(),
    UserQuestionAvgDuration(),
    Timestamp(),
    UserExpAvgCorrect(.5, .2),
    DejaVu(),
    QuestionNChoices(train),
    QuestionAnswerEntropy(train),
    UserPartCount()
]

# We filter out the extractors that have already been run]
for i, extractor in reversed(list(enumerate(extractors))):
    if pathlib.Path(f'features/{extractor}_features.csv').exists():
        extractors.pop(i)
        print(f'- Skipping {extractor}')
    else:
        print(f'- Doing {extractor}')

if not extractors:
    chime.warning()
    sys.exit('All of the features have already been extracted.')

time_taken = collections.defaultdict(lambda: {'update': 0, 'transform': 0})

# Now we loop through the training data in group order
for i, (questions, prev_group) in tqdm.tqdm(enumerate(iter_groups(train)), total=10_000):

    for extractor in extractors:

        # Update
        if isinstance(extractor, StatefulExtractor) and extractor.update_during_train:
            tic = time.time()
            extractor.update(questions, prev_group)
            time_taken[str(extractor)]['update'] += time.time() - tic

        # Transform
        tic = time.time()
        features = extractor.transform(questions)
        time_taken[str(extractor)]['transform'] += time.time() - tic

        # Sanity checks
        if isinstance(features, pd.Series):
            assert features.isnull().sum() == 0
        else:
            assert features.isnull().sum().sum() == 0

        path = f'features/{extractor}_features.csv'
        if i == 0:
            features.to_csv(path)
        else:
            features.to_csv(path, mode='a', header=False)

# Display time taken by each extractor
print(
    pd.DataFrame.from_dict(time_taken, orient='index')
    .sort_values('update')
    .applymap(lambda x: str(dt.timedelta(seconds=x)))
)

# We save the extractors so that we can reuse them during the testing phase
for extractor in extractors:
    with open(f'features/{extractor}_extractor.pkl', 'wb') as f:
        pickle.dump(extractor, f)

# In order to be able to load an extractor, we also need its source code, so we save the source
# code of each extractor
with open('features/module.py', 'w') as module:
    print('import abc\nimport pandas as pd\n', file=module)
    for obj in list(locals().values()):
        if inspect.isclass(obj) and issubclass(obj, Extractor):
            print(inspect.getsource(obj), end='\n', file=module)

chime.success()
