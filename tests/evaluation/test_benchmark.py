from unittest.mock import create_autospec
from surprise import AlgoBase, Trainset


from evaluation.benchmark import fit_and_score
from evaluation.strategies import TestMeasureStrategy, TrainMeasureStrategy


class TestFitAndScore:

    def test_fit_and_score_no_mearures(self):

        recommender_system = create_autospec(AlgoBase)
        trainset = create_autospec(Trainset)
        testset = [("user1", "item1", 5.0), ("user2", "item2", 4.0)]
        test_measures = []
        train_measures = []

        test_measurements, train_measurements, fit_time, test_time = fit_and_score(
            recommender_system, trainset, testset, test_measures, train_measures
        )

        recommender_system.fit.assert_called_once_with(trainset)
        recommender_system.test.assert_called_once_with(testset)
        assert not test_measurements
        assert not train_measurements
        assert fit_time > 0
        assert test_time > 0

    def test_fit_and_score_with_measures(self):

        recommender_system = create_autospec(AlgoBase)
        trainset = create_autospec(Trainset)
        testset = [("user1", "item1", 5.0), ("user2", "item2", 4.0)]
        test_measure = create_autospec(TestMeasureStrategy)
        train_measure = create_autospec(TrainMeasureStrategy)
        test_measures = [test_measure]
        train_measures = [train_measure]

        test_measure.calculate.return_value = 0.5
        test_measure.get_name.return_value = "test_measure"
        train_measure.calculate.return_value = 0.6
        train_measure.get_name.return_value = "train_measure"

        recommender_system.test.return_value = "predictions"

        test_measurements, train_measurements, fit_time, test_time = fit_and_score(
            recommender_system, trainset, testset, test_measures, train_measures
        )

        recommender_system.fit.assert_called_once_with(trainset)
        recommender_system.test.assert_called_once_with(testset)
        test_measure.calculate.assert_called_once_with("predictions")
        train_measure.calculate.assert_called_once_with(recommender_system)
        assert test_measurements == {test_measure.get_name.return_value: 0.5}
        assert train_measurements == {train_measure.get_name.return_value: 0.6}
        assert fit_time > 0
        assert test_time > 0

    def test_fit_and_score_with_multiple_measures(self):

        recommender_system = create_autospec(AlgoBase)
        trainset = create_autospec(Trainset)
        testset = [("user1", "item1", 5.0), ("user2", "item2", 4.0)]
        test_measure1 = create_autospec(TestMeasureStrategy)
        test_measure2 = create_autospec(TestMeasureStrategy)
        train_measure1 = create_autospec(TrainMeasureStrategy)
        train_measure2 = create_autospec(TrainMeasureStrategy)
        test_measures = [test_measure1, test_measure2]
        train_measures = [train_measure1, train_measure2]

        test_measure1.calculate.return_value = 0.5
        test_measure1.get_name.return_value = "test_measure1"
        test_measure2.calculate.return_value = 0.6
        test_measure2.get_name.return_value = "test_measure2"
        train_measure1.calculate.return_value = 0.7
        train_measure1.get_name.return_value = "train_measure1"
        train_measure2.calculate.return_value = 0.8
        train_measure2.get_name.return_value = "train_measure2"

        recommender_system.test.return_value = "predictions"

        test_measurements, train_measurements, fit_time, test_time = fit_and_score(
            recommender_system, trainset, testset, test_measures, train_measures
        )

        recommender_system.fit.assert_called_once_with(trainset)
        recommender_system.test.assert_called_once_with(testset)
        test_measure1.calculate.assert_called_once_with("predictions")
        test_measure2.calculate.assert_called_once_with("predictions")
        train_measure1.calculate.assert_called_once_with(recommender_system)
        train_measure2.calculate.assert_called_once_with(recommender_system)
        assert test_measurements == {
            test_measure1.get_name.return_value: 0.5,
            test_measure2.get_name.return_value: 0.6,
        }
        assert train_measurements == {
            train_measure1.get_name.return_value: 0.7,
            train_measure2.get_name.return_value: 0.8,
        }
        assert fit_time > 0
        assert test_time > 0
