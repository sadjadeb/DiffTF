from os import path
from scipy import sparse
import pandas
from collections import Counter
import numpy as np
import pickle as pkl
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import word_tokenize, RegexpTokenizer
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
import pandas as pd
# from ml.team2vec import *
import matplotlib.pyplot as plt
import xlwt
import eval.evaluator as dblp_eval
from tempfile import TemporaryFile
from iteration_utilities import deepflatten


publication_filter = ['sigmod', 'vldb', 'icde', 'icdt', 'edbt', 'pods', 'kdd', 'www',
                      'sdm', 'pkdd', 'icdm', 'cikm', 'aaai', 'icml', 'ecml', 'colt',
                      'uai', 'soda', 'focs', 'stoc', 'stacs']
'''
currentArticle.venue.toLowerCase().contains("sigmod") ||
                                currentArticle.venue.toLowerCase().contains("vldb") ||
                                currentArticle.venue.toLowerCase().contains("icde") ||
                                currentArticle.venue.toLowerCase().contains("icdt") ||
                                currentArticle.venue.toLowerCase().contains("edbt") ||
                                currentArticle.venue.toLowerCase().contains("pods") ||
                                currentArticle.venue.toLowerCase().contains("kdd") ||
                                currentArticle.venue.toLowerCase().contains("www") ||
                                currentArticle.venue.toLowerCase().contains("sdm") ||
                                currentArticle.venue.toLowerCase().contains("pkdd") ||
                                currentArticle.venue.toLowerCase().contains("icdm") ||
                                currentArticle.venue.toLowerCase().contains("cikm") ||
                                currentArticle.venue.toLowerCase().contains("aaai") ||
                                currentArticle.venue.toLowerCase().contains("icml") ||
                                currentArticle.venue.toLowerCase().contains("ecml") ||
                                currentArticle.venue.toLowerCase().contains("colt") ||
                                currentArticle.venue.toLowerCase().contains("uai") ||
                                currentArticle.venue.toLowerCase().contains("soda") ||
                                currentArticle.venue.toLowerCase().contains("focs") ||
                                currentArticle.venue.toLowerCase().contains("stoc") ||
                                currentArticle.venue.toLowerCase().contains("stacs")
//                                currentArticle.venue.toLowerCase().contains("data") ||
//                                currentArticle.venue.toLowerCase().contains("knowledge")
//                                currentArticle.venue.toLowerCase().contains("min") ||
//                                currentArticle.venue.toLowerCase().contains("computer") ||
//                                currentArticle.venue.toLowerCase().contains("system") ||
//                                currentArticle.venue.toLowerCase().contains("anal") ||
//                                currentArticle.venue.toLowerCase().contains("network") ||
//                                currentArticle.venue.toLowerCase().contains("proc") ||
//                                currentArticle.venue.toLowerCase().contains("design") ||
//                                currentArticle.venue.toLowerCase().contains("inter") ||
//                                currentArticle.venue.toLowerCase().contains("exp") ||
//                                currentArticle.venue.toLowerCase().contains("theory") ||
'''


def load_dblp_arnet(infname, outfname, ftype='dict'):  # source: https://gist.github.com/cntswj/51d3379692fd5e553cb6
    # dictionary version added to it
    if ftype == 'dict':
        with open(infname, 'r', encoding='utf-8') as f:
            output = []
            count = 0
            for key, group in groupby(f, key=lambda l: l.strip(' \n\r') == ''):
                if not key:
                    refs = []
                    authors = []
                    title, venue, year, idx, abstract = [''] * 5
                    for item in group:
                        item = item.strip(' \r\n')
                        if item.startswith('#*'):
                            title = item[2:]
                        elif item.startswith('#@'):
                            authors = item[2:].split(',')
                        elif item.startswith('#t'):
                            year = item[2:]
                        elif item.startswith('#c'):
                            venue = item[2:]
                        elif item.startswith('#index'):
                            idx = item[6:]
                        elif item.startswith('#!'):
                            abstract = item[2:]
                        elif item.startswith('#%'):
                            refs.append(item[2:])
                    output.append(
                        {'idx': idx, 'title': title, 'venue': venue, 'authors': authors, 'year': year, 'refs': refs,
                         'abstract': abstract})
                    count += 1
                    print('\r%d\tlines' % (count,), )
        with open(outfname, 'wb') as f:
            pkl.dump(output, f)
    elif ftype == 'csv':
        with open(infname, 'r', encoding='utf-8') as f, open(outfname, 'w', encoding='utf-8') as csvfile:
            csv_writer = csv.writer(
                csvfile, delimiter=',',
                quotechar='|', quoting=csv.QUOTE_MINIMAL)
            count = 0
            for key, group in groupby(f, key=lambda l: l.strip(' \n\r') == ''):
                if not key:
                    refs = []
                    authors = []
                    title, venue, year, idx, abstract = [''] * 5
                    for item in group:
                        item = item.strip(' \r\n')
                        if item.startswith('#*'):
                            title = item[2:]
                        elif item.startswith('#@'):
                            authors = item[2:].split(',')
                        elif item.startswith('#t'):
                            year = item[2:]
                        elif item.startswith('#c'):
                            venue = item[2:]
                        elif item.startswith('#index'):
                            idx = item[6:]
                        elif item.startswith('#!'):
                            abstract = item[2:]
                        elif item.startswith('#%'):
                            refs.append(item[2:])
                    csv_writer.writerow(
                        [idx, title, venue, authors, year, refs, abstract])
                    count += 1
                    print('\r%d\tlines' % (count,), )


def load_citation_csv(input_file):
    data = np.recfromcsv(input_file, encoding='utf_8')
    return data


def load_citation_pkl(input_file):
    with open(input_file, 'rb') as pickle_file:
        data = pkl.load(pickle_file)
    return data


def load_skills(dir):
    skills_counts = pandas.read_csv(dir, encoding='utf_8', header=None, delimiter='	')
    skills = skills_counts.iloc[:, 0]
    skills_frequency = skills_counts.iloc[:, 1]
    return skills, skills_frequency


def load_authors(dir):
    authorNameIds = pandas.read_csv(dir, encoding='utf_8', header=None, delimiter='	', names=["NameID", "Author"])
    authorNameIds_sorted = authorNameIds.sort_values(by='NameID')
    nameIDs = authorNameIds_sorted.iloc[:, 0]
    authors = authorNameIds_sorted.iloc[:, 1]
    return authors.values, nameIDs.values


def get_user_skill_dict(data):
    dict = {}
    for sample in data:
        id = sample[0]
        skill = sample[1].nonzero()[1]
        user = sample[2].nonzero()[1]
        for u in user:
            if u not in dict.keys():
                dict[u] = []
            dict[u].extend(skill)
    return dict


def get_foldIDsampleID_stata_dict(data, train_test_indices, kfold=10):
    evaluation_k_set = np.arange(1, kfold + 1, 1)
    foldIDsampleID_stata_dict = dblp_eval.init_eval_holder(evaluation_k_set)
    for fold_counter in evaluation_k_set:
        _, _, x_test, _ = get_fold_data(fold_counter, data, train_test_indices, mute=True)
        for smaple in x_test:
            #print(f"DEBUG: sample[0] shape: {smaple[0].shape}")
            #print(f"DEBUG: sample[0] nonzero: {smaple[0].nonzero()}")
            foldIDsampleID_stata_dict[fold_counter].append(len(smaple[0].nonzero()[1]))
    return foldIDsampleID_stata_dict


def convert_to_pkl(txt_dir='dataset/dblp.txt', pkl_dir='dataset/dblp.pkl', ftype='dict'):
    load_dblp_arnet(txt_dir, pkl_dir, ftype=ftype)


# dblp to sparse matrix: output: pickle file of the sparse matrix
def extract_data(filter_journals=True, size_limit=np.inf, skill_size_filter=0, member_size_filter=0,
                 source_dir='dataset/dblp.pkl', skill_dir='dataset/invertedTermCount.txt',
                 author_dir='dataset/authorNameId.txt', output_dir='dataset/ae_dataset.pkl', documentID_venue_dir='dataset/documentID_venue.pkl'):
    if not source_pkl_exist(file_path=source_dir):
        convert_to_pkl()

    data = load_citation_pkl(source_dir)
    skills, skills_freq = load_skills(skill_dir)
    authors, _ = load_authors(author_dir)
    skills = np.asarray(skills)
    authors = [author.strip().lower() for author in authors]

    dataset = []
    documentID_venue = []
    counter = 0
    id = 0
    for record in data:
        id += 1
        if filter_journals and not filter_pubs(record['venue']):
            continue
        skill_vector = np.zeros(skills.__len__())
        user_vector = np.zeros(authors.__len__())

        for author in record['authors']:
            try:
                user_vector[authors.index(author.strip().lower())] = 1
            except:
                pass

        for i, s in enumerate(skills):
            if s in record['title']:
                skill_vector[i] = 1

        if np.sum(skill_vector) <= skill_size_filter or np.sum(user_vector) <= member_size_filter:
            continue

        skill_vector_sparse = sparse.coo_matrix(skill_vector)
        user_vector_sparse = sparse.coo_matrix(user_vector)
        dataset.append([id, skill_vector_sparse, user_vector_sparse])
        documentID_venue.append([id, record['venue']])
        print("File {} added to dataset file. (Total Files: {})".format(id, counter + 1))

        counter += 1
        if counter >= size_limit:
            break

    with open(output_dir, 'wb') as f:
        pkl.dump(dataset, f)
        print('{} records saved to {} successfully.'.format(counter + 1, output_dir))
    
    with open(documentID_venue_dir, 'wb') as f:
        pkl.dump(documentID_venue, f)
        print('{} document_venue pairs saved to {} successfully.'.format(counter + 1, documentID_venue_dir))


def load_ae_dataset(file_path='dataset/ae_dataset.pkl'):
    with open(file_path, 'rb') as f:
        dataset = pkl.load(f)
    return dataset


# we should check if we have the proper dataset for ae or not. and if not then we run extract_data function
def ae_data_exist(file_path='dataset/ae_dataset.pkl'):
    if path.exists(file_path):
        return True
    else:
        return False


# we should check if we have the source dataset or not. and if not then we run convert_to_pkl() function
def source_pkl_exist(file_path='dataset/dblp.pkl'):
    if path.exists(file_path):
        return True
    else:
        return False


# Filter wanted publications by their journal
def filter_pubs(venue: str):
    found = False
    for allowed_pub in publication_filter:
        if allowed_pub.lower() in venue.lower():
            found = True
            break
    return found


def nn_t2v_dataset_generator(model, dataset, output_file_path, mode='user'):
    t2v_dataset = []
    counter = 1
    for record in dataset:
        id = record[0]
        if mode.lower() == 'user':
            try:
                skill_vec = record[1].todense()
                team_vec = model.get_team_vec(id)
                t2v_dataset.append([id, skill_vec, team_vec])
                print('Record #{} | File #{} appended to dataset.'.format(counter, id))
                counter += 1
            except:
                print('Cannot add record with id {}'.format(id))
        elif mode.lower() == 'skill':
            try:
                skill_vec = model.get_team_vec(id)
                team_vec = record[2].todense()
                t2v_dataset.append([id, skill_vec, team_vec])
                print('Record #{} | File #{} appended to dataset.'.format(counter, id))
                counter += 1
            except:
                print('Cannot add record with id {}'.format(id))
        elif mode.lower() == 'full':
            try:
                model_skill = model['skill']
                model_user = model['user']
                skill_vec = model_skill.get_team_vec(id)
                team_vec = model_user.get_team_vec(id)
                t2v_dataset.append([id, skill_vec, team_vec])
                print('Record #{} | File #{} appended to dataset.'.format(counter, id))
                counter += 1
            except:
                print('Cannot add record with id {}'.format(id))
        elif mode.lower() == 'uu':
            try:
                team_dense = record[2].todense()
                team_vec = model.get_team_vec(id)
                t2v_dataset.append([id, team_vec, team_dense])
                print('Record #{} | File #{} appended to dataset.'.format(counter, id))
                counter += 1
            except:
                print('Cannot add record with id {}'.format(id))
    with open(output_file_path, 'wb') as f:
        pkl.dump(t2v_dataset, f)


def nn_h2v_dataset_generator(model_path='dataset/embedding_hinsage_dict.pkl', dataset=None, output_file_path='dataset/h2v.pkl', mode='user'):
    model = pkl.load(open(model_path, 'rb'))
    h2v_dataset = []
    counter = 1
    for record in dataset:
        id = record[0]
        if mode.lower() == 'user':
            try:
                skill_vec = record[1].todense()
                team_vec = model[id]
                h2v_dataset.append([id, skill_vec, team_vec])
                print('Record #{} | File #{} appended to dataset.'.format(counter, id))
                counter += 1
            except:
                print('Cannot add record with id {}'.format(id))
        elif mode.lower() == 'skill':
            try:
                skill_vec = model[id]
                team_vec = record[2].todense()
                h2v_dataset.append([id, skill_vec, team_vec])
                print('Record #{} | File #{} appended to dataset.'.format(counter, id))
                counter += 1
            except:
                print('Cannot add record with id {}'.format(id))
        elif mode.lower() == 'full':
            try:
                skill_vec = model[id]
                team_vec = model[id]
                h2v_dataset.append([id, skill_vec, team_vec])
                print('Record #{} | File #{} appended to dataset.'.format(counter, id))
                counter += 1
            except:
                print('Cannot add record with id {}'.format(id))
        elif mode.lower() == 'uu':
            try:
                team_dense = record[2].todense()
                team_vec = model[id]
                h2v_dataset.append([id, team_vec, team_dense])
                print('Record #{} | File #{} appended to dataset.'.format(counter, id))
                counter += 1
            except:
                print('Cannot add record with id {}'.format(id))
    with open(output_file_path, 'wb') as f:
        pkl.dump(h2v_dataset, f)


def nn_m2v_dataset_generator(model_path='dataset/embedding_dict.pkl', dataset=None, output_file_path='dataset/m2v.pkl', mode='user'):
    model = pkl.load(open(model_path, 'rb'))
    m2v_dataset = []
    counter = 1
    for record in dataset:
        id = record[0]
        if mode.lower() == 'user':
            try:
                skill_vec = record[1].todense()
                team_idx = record[2].todense().nonzero()[1]
                team_vec = []
                for user_id in team_idx:
                    print()
                    team_vec.append(model['user'][user_id])
                m2v_dataset.append([id, skill_vec, np.sum(team_vec, axis=0)])
                print('Record #{} | File #{} appended to dataset.'.format(counter, id))
                counter += 1
            except:
                print('Cannot add record with id {}'.format(id))
        elif mode.lower() == 'skill':
            try:
                skill_idx = record[1].todense().nonzero()[1]
                skill_vec = []
                for skill_id in skill_idx:
                    skill_vec.append(model['skill'][skill_id])
                team_vec = record[2].todense()
                m2v_dataset.append([id, np.sum(skill_vec, axis=0), team_vec])
                print('Record #{} | File #{} appended to dataset.'.format(counter, id))
                counter += 1
            except:
                print('Cannot add record with id {}'.format(id))
        elif mode.lower() == 'full':
            try:
                skill_idx = record[1].todense().nonzero()[1]
                skill_vec = []
                for skill_id in skill_idx:
                    skill_vec.append(model['skill'][skill_id])
                team_idx = record[2].todense().nonzero()[1]
                team_vec = []
                for user_id in team_idx:
                    team_vec.append(model['user'][user_id])
                m2v_dataset.append([id, np.sum(skill_vec, axis=0), np.sum(team_vec, axis=0)])
                print('Record #{} | File #{} appended to dataset.'.format(counter, id))
                counter += 1
            except:
                print('Cannot add record with id {}'.format(id))
        # elif mode.lower() == 'uu':
        #     try:
        #         team_dense = record[2].todense()
        #         team_vec = model.get_team_vec(id)
        #         m2v_dataset.append([id, team_vec, team_dense])
        #         print('Record #{} | File #{} appended to dataset.'.format(counter, id))
        #         counter += 1
        #     except:
        #         print('Cannot add record with id {}'.format(id))
    with open(output_file_path, 'wb') as f:
        pkl.dump(m2v_dataset, f)


def nn_m2v_embedding_dataset_generator(model_path='dataset/embedding_dict.pkl', dataset=None, output_file_path='dataset/e_m2v.pkl', mode='user', max_length=22):
    model = pkl.load(open(model_path, 'rb'))
    m2v_dataset = []
    counter = 1
    for record in dataset:
        id = record[0]
        if mode.lower() == 'user':
            try:
                skill_vec = record[1].todense()
                team_idx = record[2].todense().nonzero()[1]
                team_vec = []
                for user_id in team_idx:
                    print()
                    team_vec.append(model['user'][user_id])
                m2v_dataset.append([id, skill_vec, np.sum(team_vec, axis=0)])
                print('Record #{} | File #{} appended to dataset.'.format(counter, id))
                counter += 1
            except:
                print('Cannot add record with id {}'.format(id))
        elif mode.lower() == 'skill':
            try:
                skill_idx = record[1].todense().nonzero()[1]
                padding = max_length - len(skill_idx)
                skill_vec = list(skill_idx) + [0]*padding
                team_vec = record[2].todense()
                m2v_dataset.append([id, skill_vec[0], team_vec])
                print('Record #{} | File #{} appended to dataset.'.format(counter, id))
                counter += 1
            except:
                print('Cannot add record with id {}'.format(id))
        elif mode.lower() == 'full':
            try:
                skill_idx = record[1].todense().nonzero()[1]
                skill_vec = []
                for skill_id in skill_idx:
                    skill_vec.append(model['skill'][skill_id])
                team_idx = record[2].todense().nonzero()[1]
                team_vec = []
                for user_id in team_idx:
                    team_vec.append(model['user'][user_id])
                m2v_dataset.append([id, np.sum(skill_vec, axis=0), np.sum(team_vec, axis=0)])
                print('Record #{} | File #{} appended to dataset.'.format(counter, id))
                counter += 1
            except:
                print('Cannot add record with id {}'.format(id))
        # elif mode.lower() == 'uu':
        #     try:
        #         team_dense = record[2].todense()
        #         team_vec = model.get_team_vec(id)
        #         m2v_dataset.append([id, team_vec, team_dense])
        #         print('Record #{} | File #{} appended to dataset.'.format(counter, id))
        #         counter += 1
        #     except:
        #         print('Cannot add record with id {}'.format(id))
    with open(output_file_path, 'wb') as f:
        pkl.dump(m2v_dataset, f)


def get_memebrID_by_teamID(preds_ids):
    dataset = load_preprocessed_dataset()
    dataset = pandas.DataFrame(dataset, columns=['id', 'skill', 'author'])
    preds_authors_ids = []
    for pred_ids in preds_ids:
        authors_ids = []
        for id in pred_ids:
            try:
                found_record = dataset.loc[dataset['id'] == id]['author']
                sparse_authors = sparse.coo_matrix(found_record.values.all())
                author_ids = sparse_authors.nonzero()[1]
                authors_ids.extend(author_ids)
            except:
                print('Cannot find team for sample with id: {}'.format(id))
        preds_authors_ids.append(authors_ids)
    return preds_authors_ids


def tokenize(text):
    tokenizer = RegexpTokenizer(r'\w+')
    tokens = tokenizer.tokenize(text)
    tokens = [word for word in tokens if word.isalpha()]
    stemmer = PorterStemmer()
    tokens = [stemmer.stem(word) for word in tokens]
    return tokens


def dataset_preprocessing(dataset, min_records=10, kfolds=10, max_features=2000, n_gram=3,
                          dataset_source_dir='dataset/dblp.pkl', save_to_pkl=True, save_to_csv=True,
                          author_dir='dataset/authorNameId.txt',
                          indices_dict_file_path='dataset/Train_Test_indices.pkl', baseline_path='dataset/',
                          preprocessed_dataset_file_path='dataset/dblp_preprocessed_dataset.pkl', seed=7):
    random.seed(seed)
    author_paper_counter = Counter()
    author_docID_dict = {}
    docID_author_dict = {}

    for sample in dataset:
        id = sample[0]
        sparse_authorIDs = sparse.coo_matrix(sample[2])
        author_ids = sparse_authorIDs.nonzero()[1]
        for author_id in author_ids:
            if author_id not in author_docID_dict.keys():
                author_docID_dict[author_id] = []
            author_docID_dict[author_id].append(id)
        author_paper_counter.update(author_ids)

    eligible_authors = [x for x in author_paper_counter.keys() if author_paper_counter.get(x) >= min_records]

    eligible_documents = []
    for eligible_author in eligible_authors:
        eligible_documents.extend(author_docID_dict[eligible_author])
    eligible_documents = np.unique(eligible_documents).tolist()

    data = load_citation_pkl(dataset_source_dir)
    data = np.asarray(data)
    eligible_titles = []
    for eligible_paper in data[eligible_documents]:
        eligible_titles.append(eligible_paper['title'].strip())
    vect = TfidfVectorizer(tokenizer=tokenize, analyzer='word', lowercase=True, stop_words=stopwords.words('english'),
                           ngram_range=(1, n_gram), max_features=max_features)
    vect.fit(eligible_titles)

    for sample in dataset:
        id = sample[0]
        if id in eligible_documents:
            sparse_authorIDs = sparse.coo_matrix(sample[2])
            author_ids = sparse_authorIDs.nonzero()[1]
            docID_author_dict[id] = author_ids

    author_set = []
    skill_sets = []
    id_sets = []
    for eligible_document_id in eligible_documents:
        skill_set = vect.transform([data[eligible_document_id]['title'].strip()])
        skill_set = skill_set.tocoo()
        if skill_set.count_nonzero() > 0:
            author_set.extend(docID_author_dict[eligible_document_id])
            id_sets.append(eligible_document_id)
            skill_sets.append(skill_set)
        else:
            eligible_documents.remove(eligible_document_id)
            affected_authors = docID_author_dict[eligible_document_id]
            del docID_author_dict[eligible_document_id]
            for author_docID_dict_key in affected_authors:
                if eligible_document_id in author_docID_dict[author_docID_dict_key]:
                    author_docID_dict[author_docID_dict_key].remove(eligible_document_id)
    author_set = np.unique(author_set).tolist()

    print('Number of eligible documents equal/more than {} records: {}'.format(min_records, len(eligible_documents)))
    print('Number of eligible authors equal/more than {} records: {}'.format(min_records, len(eligible_authors)))

    eligible_documents = []
    preprocessed_dataset = []
    for id, skill_vector in zip(id_sets, skill_sets):
        author_vector = np.zeros(len(eligible_authors))
        for author_id in docID_author_dict[id]:
            if author_id in eligible_authors:
                author_vector[eligible_authors.index(author_id)] = 1
        author_vector = sparse.coo_matrix(author_vector)
        if len(skill_vector.nonzero()[1]) > 1 and len(author_vector.nonzero()[1]) > 1:
            preprocessed_dataset.append([id, skill_vector, author_vector])
            eligible_documents.append(id)

    if save_to_pkl:
        with open('{}'.format(preprocessed_dataset_file_path), 'wb') as f:
            pkl.dump(preprocessed_dataset, f)

    indices = split_data(kfolds, author_docID_dict, eligible_documents, save_to_pkl, save_to_csv,
                         indices_dict_file_path, baseline_path)

    with open('{}preprocessed_authorNameId.txt'.format(baseline_path), 'w') as f:
        authorNames, _ = load_authors(author_dir)
        for idx, eligible_author in enumerate(eligible_authors):
            line = [str(idx), authorNames[eligible_author]]
            f.write('\t'.join(line) + '\n')
        f.close()

    for fold in range(1, kfolds+1):
        train_docs = indices[fold]['Train']
        test_docs = indices[fold]['Test']
        authorNames, _ = load_authors(author_dir)
        authorNames = [author.strip().lower() for author in authorNames]

        feature_names = np.asarray(vect.get_feature_names())
        if save_to_csv:
            with open('{}baseline_authorName_skill_train_{}.csv'.format(baseline_path, fold), 'w') as f:
                for eligible_author in eligible_authors:
                    docIDs = author_docID_dict[eligible_author]
                    docIDs = [docID for docID in docIDs if docID in train_docs]
                    docsTitles = [d['title'].strip() for d in data[docIDs]]
                    features = Counter()
                    for title in docsTitles:
                        features.update(feature_names[vect.transform([title]).nonzero()[1]])
                    line = [authorNames[eligible_author]]
                    line.extend(features.keys())
                    f.write(','.join(line) + '\n')
                f.close()
            with open('{}baseline_skill_test_{}.csv'.format(baseline_path, fold), 'w') as f:
                docsTitles = [d['title'].strip() for d in data[test_docs]]
                for docTitle in docsTitles:
                    line = feature_names[vect.transform([docTitle]).nonzero()[1]]
                    f.write(','.join(line) + '\n')
                f.close()

    return indices, preprocessed_dataset


def split_data(kfolds, author_docID_dict, eligible_documents, save_to_pkl, save_to_csv,
               indices_dict_file_path, baseline_path):

    indices = {}
    for fold_counter in range(1, kfolds+1):
        train_docs = []
        test_docs = []
        rule_violence_counter = 0

        for ex_folds in range(1, fold_counter):
            train_docs.extend(indices[ex_folds]['Test'])

        train_docs = list(deepflatten(train_docs))
        c_train = Counter()
        c_train.update(train_docs)
        train_docs = list(c_train.keys())
        train_docs.sort()

        for author in author_docID_dict.keys():
            list_of_author_docs = author_docID_dict.get(author)
            list_of_author_docs = [docID for docID in list_of_author_docs if docID in eligible_documents]

            number_of_test_docs = np.ceil((1 / kfolds) * len(list_of_author_docs))

            already_in_test = []
            for doc in list_of_author_docs:
                if doc in test_docs:
                    already_in_test.append(doc)

            number_of_moving_to_test = int(number_of_test_docs - len(already_in_test))

            if number_of_moving_to_test <= 0:
                rule_violence_counter += 1

            elif number_of_moving_to_test > 0:
                eligible_samples_for_test_set = [doc for doc in list_of_author_docs if
                                                 doc not in already_in_test and doc not in train_docs]
                if len(eligible_samples_for_test_set) > 0:
                    if len(eligible_samples_for_test_set) > number_of_moving_to_test:
                        test_docs.extend(random.sample(eligible_samples_for_test_set, k=number_of_moving_to_test))
                    else:
                        test_docs.extend(eligible_samples_for_test_set)

            train_docs.extend([ele for ele in list_of_author_docs if ele not in test_docs and ele not in train_docs])

        print("******* Fold {} *******".format(fold_counter))
        print("Number of Train docs: {}".format(len(train_docs)))
        print("Number of Test docs: {}".format(len(test_docs)))
        print("Number of violations in train/test split because of already"
              " a paper from target author in test set: {} exists.".format(rule_violence_counter))

        test_docs = list(deepflatten(test_docs))
        c_test = Counter()
        c_test.update(test_docs)
        test_docs = list(c_test.keys())
        test_docs.sort()

        indices[fold_counter] = {'Train': train_docs, 'Test': test_docs}

        if save_to_csv:
            with open('{}baseline_train_indices_{}.csv'.format(baseline_path, fold_counter), 'w') as f:
                f.write(','.join(str(x) for x in train_docs) + '\n')
                f.close()
            with open('{}baseline_test_indices_{}.csv'.format(baseline_path, fold_counter), 'w') as f:
                f.write(','.join(str(x) for x in test_docs) + '\n')
                f.close()

    if save_to_pkl:
        with open('{}'.format(indices_dict_file_path), 'wb') as f:
            pkl.dump(indices, f)

    return indices

import os
import torch

def get_fold_data(fold_counter, dataset, train_test_indices, mute=False):
    x_train = []
    y_train = []
    x_test = []
    y_test = []
    expert_save = []
    skill_save = []
    train_index = train_test_indices[fold_counter]['Train']
    #print(train_index[0])
    test_index = train_test_indices[fold_counter]['Test']
    for sample in dataset:
        id = sample[0]
        if id in train_index:
            x_train.append(sample[1])
            y_train.append(sample[2])
        elif id in test_index:
            x_test.append(sample[1])
            y_test.append(sample[2])
        skill_save.append(sample[1])
        expert_save.append(sample[2])


    #expert_noisy = expert_save + (torch.randn_like(expert_save)* 0.010725 + 0.025185)
    x_train = np.asarray(x_train).reshape(len(x_train), -1)
    y_train = np.asarray(y_train).reshape(len(y_train), -1)
    x_test = np.asarray(x_test).reshape(len(x_test), -1)
    y_test = np.asarray(y_test).reshape(len(y_test), -1)
    expert_save = np.asarray(expert_save).reshape(len(expert_save), -1)
    skill_save = np.asarray(skill_save).reshape(len(skill_save), -1)
    #expert_noisy = np.asarray(expert_noisy).reshape(len(expert_noisy), -1)
    #expert_np = expert_save
    #expert_noisy = expert_np + (torch.randn_like(torch.from_numpy(expert_np))* 0.010725 + 0.025185)
    #if not os.path.exists('./output/embeddings'):
    #        os.makedirs('./output/embeddings')
    #with open('./output/embeddings/expert_noisy.npy', 'ab') as f:
    #        np.save(f, expert_noisy)

    if not mute:
        print('Fold number {}'.format(fold_counter))
        print('dataset Size: {}'.format(len(dataset)))
        print('Train Size: {} Test Size: {}'.format(x_train.__len__(), x_test.__len__()))
    return x_train, y_train, x_test, y_test#, expert_save, skill_save


def train_test_indices_exist(file_path='dataset/Train_Test_indices.pkl'):
    if path.exists(file_path):
        return True
    return False


def load_train_test_indices(file_path='dataset/Train_Test_indices.pkl'):
    with open(file_path, 'rb') as f:
        indices = pkl.load(f)
    return indices


def load_performance_feedback(file_path):
    with open(file_path, 'rb') as f:
        feedback = pkl.load(f)
    return feedback


def preprocessed_dataset_exist(file_path='dataset/dblp_preprocessed_dataset.pkl'):
    if path.exists(file_path):
        return True
    return False


def load_preprocessed_dataset(file_path='dataset/dblp_preprocessed_dataset.pkl'):
    with open(file_path, 'rb') as f:
        dataset = pkl.load(f)
    return dataset


def create_user_item(x, y):
    itemID = []
    userID = []
    rating = []
    for skills, users in zip(x, y):
        skills_ids = list(skills.all().nonzero()[1])
        users_ids = list(users.all().nonzero()[1])
        for skill in skills_ids:
            for user in users_ids:
                itemID.append(skill)
                userID.append(user)
                rating.append(1)
    df = pd.DataFrame({'itemID': itemID, 'userID': userID, 'rating': rating})
    return df


def get_user_HIndex(file_path='dataset/authorHIndex.txt'):
    user_hindex_dict = {}
    user_hindex = pandas.read_csv(file_path, encoding='utf_8', header=None, delimiter='	')
    user_hindex = (user_hindex.iloc[:, :2]).values
    for item in user_hindex:
        user_hindex_dict[str(int(item[0])-1)] = item[1]
    return user_hindex_dict


# def get_preprocessed_authorNameID_dict(file_path='../dataset/preprocessed_authorNameId.txt'):
def get_preprocessed_authorNameID_dict(file_path='dataset/authorNameId.txt'):
    authorNames = load_authors(file_path)
    authorID = {x: i.strip().lower() for i, x in zip(authorNames[0], authorNames[1])}
    return authorID

def get_preprocessed_dotaAuthorNameID_dict(file_path='dataset/authorNameId.txt'):
    authorNames = load_authors(file_path)
    authorID = {x: i for i, x in zip(authorNames[0], authorNames[1])}
    return authorID


def dataset_histo(min_count=3):
    data = load_preprocessed_dataset()
    user_count = []
    skill_count = []
    for sample in data:
        if len(sample[2].nonzero()[1]) > min_count:
            skill_count.append(len(sample[1].nonzero()[1]))
            user_count.append(len(sample[2].nonzero()[1]))

    plt.figure(0)
    plt.hist(skill_count, bins='auto')
    skill_hist, skill_bins = np.histogram(skill_count, bins=range(30))

    plt.figure(1)
    plt.hist(user_count, bins='auto')
    user_hist, user_bins = np.histogram(user_count, bins=range(20))

    book = xlwt.Workbook()
    sheet1 = book.add_sheet('skill')
    sheet2 = book.add_sheet('user')

    sheet1.write(0,0,'bins')
    sheet1.write(0,1,'quantity')
    for i, e in enumerate(skill_hist):
        sheet1.write(i+1, 0, int(skill_bins[i]))
        sheet1.write(i+1, 1, int(skill_hist[i]))

    sheet2.write(0,0,'bins')
    sheet2.write(0,1,'quantity')
    for i, e in enumerate(user_hist):
        sheet2.write(i+1, 0, int(user_bins[i]))
        sheet2.write(i+1, 1, int(user_hist[i]))

    name = "histogram.xls"
    book.save(name)
    book.save(TemporaryFile())

    plt.show()

    return np.mean(skill_count), np.mean(user_count)

# dataset_histo(min_count=0)


def get_co_occurrence(save_to_file=True):
    data = load_preprocessed_dataset('./dataset/dblp_preprocessed_dataset.pkl')
    mat = np.zeros((data[0][2].shape[1], data[0][2].shape[1]))
    for sample in data:
        indices = sample[2].nonzero()[1]
        for i in indices:
            for j in indices:
                if i != j:
                    mat[i, j] += 1

    if save_to_file:
        np.savetxt('./output/eval_results/co_occurrence.csv', mat, delimiter=',', fmt='%d')

    return mat


def get_coauthors():
    data = load_preprocessed_dataset()
    coauthors = {}
    for sample in data:
        users = sample[2].nonzero()[1]
        for user in users:
            if user not in coauthors.keys():
                coauthors[user] = []
            coauthors_list = list(users).copy()
            coauthors_list.remove(user)
            coauthors[user].extend(coauthors_list)
    return coauthors


def dataset_aug():
    data = load_preprocessed_dataset()
    coauthors_dict = get_coauthors()
    author_skill_dict = get_user_skill_dict(data)
    skill_vector_dim = data[0][1].shape[1]
    user_vector_dim = data[0][2].shape[1]
    last_id = data[-1][0]
    synthetic_data = []
    counter = 0
    for author_counter, author in enumerate(coauthors_dict.keys()):
        print('{}/{} of authors. {} Records added.'.format(author_counter, len(coauthors_dict.keys()), counter))
        coauthors = coauthors_dict[author]
        for coauthor in coauthors:
            coauthor_coauthors = coauthors_dict[coauthor]
            first_tier_links = coauthors.copy()
            first_tier_links.remove(coauthor)
            for i in coauthor_coauthors:
                if i in first_tier_links:
                    first_tier_links.remove(i)
            if len(first_tier_links)>0:
                for j in first_tier_links:
                    #create sample
                    skill_vector = np.zeros((1, skill_vector_dim))
                    skills = []
                    skills.extend(author_skill_dict[author])
                    skills.extend(author_skill_dict[coauthor])
                    skills.extend(author_skill_dict[j])
                    for s in skills:
                        skill_vector[0, s] = 1
                    user_vector = np.zeros((1, user_vector_dim))
                    user_vector[0, author] = 1
                    user_vector[0, coauthor] = 1
                    user_vector[0, j] = 1

                    id = last_id + 1
                    sample = [id, sparse.coo_matrix(skill_vector), sparse.coo_matrix(user_vector)]
                    synthetic_data.append(sample)
                    #update indices
                    last_id += 1
                    counter += 1
                    # update coauthors
                    coauthors_dict[coauthor].append(j)
                    coauthors_dict[j].append(coauthor)

    #save synthetic dataset
    save = input('Save synthetic data?')
    if save.lower() == 'y':
        with open('dataset/synthetic_dataset.pkl', 'wb') as f:
             pkl.dump(synthetic_data, f)
        print('Saved synthetic dataset successfully.')


def get_preprocessed_user_HIndex(user_hindex_path='dataset/authorHIndex.txt', preprocessed_authorNameID_path='dataset/preprocessed_authorNameId.txt', authorNameID_path='dataset/authorNameId.txt'):
    preprocessed_authorNameID = get_preprocessed_authorNameID_dict(file_path=preprocessed_authorNameID_path)
    user_hindex_dict = {}
    user_hindex = pandas.read_csv(user_hindex_path, encoding='utf_8', header=None, delimiter='	')
    user_hindex = (user_hindex.iloc[:, :2]).values
    for item in user_hindex:
        user_hindex_dict[str(int(item[0])-1)] = item[1]
    authorNameID = load_authors(authorNameID_path)
    author_nameID = {i.strip().lower(): x for i, x in zip(authorNameID[0], authorNameID[1])}

    preprocessed_user_HIndex = {x: user_hindex_dict.get(str(author_nameID.get(preprocessed_authorNameID.get(x)))) for x in list(preprocessed_authorNameID.keys())}

    return preprocessed_user_HIndex