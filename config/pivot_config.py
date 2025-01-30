class PivotConfig:
    total_training_epoch = 64
    using_pivot_learning = True
    aux_loss_weight = 0.750

    num_of_history = 50
    num_of_future = 60
    pivot_recurrent_fusing_query = False
    pivot_recurrent_num_config = [6, 6, 6]
    pivot_refine = False
    traj_pred_recurrent = False
    trajectory_refine = True

    predict_trajectory_from_pivot_offset = True
    totally_gt_pivot = False
    teach_force_on_pivot = False


    predict_bidirection_interpolation_delta = False

    using_anchor_refine = False

    visualize = False