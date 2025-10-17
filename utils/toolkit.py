import os
import numpy as np
import torch


def count_parameters(model, trainable=False):
    if trainable:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())


def tensor2numpy(x):
    if isinstance(x, torch.Tensor):
        return x.cpu().data.numpy() if x.is_cuda else x.data.numpy()
    elif isinstance(x, (int, float, np.integer, np.floating)):
        return np.array(x)
    else:
        raise TypeError("Input must be a torch.Tensor, int, or float.")


def target2onehot(targets, n_classes):
    onehot = torch.zeros(targets.shape[0], n_classes).to(targets.device)
    onehot.scatter_(dim=1, index=targets.long().view(-1, 1), value=1.0)
    return onehot


def makedirs(path):
    if not os.path.exists(path):
        os.makedirs(path)


def accuracy(y_pred, y_true, nb_old, increment=10):
    assert len(y_pred) == len(y_true), "Data length error."
    all_acc = {}
    all_acc["total"] = np.around((y_pred == y_true).sum() * 100 / len(y_true), decimals=2)

    # Grouped accuracy
    for class_id in range(0, np.max(y_true), increment):
        idxes = np.where(np.logical_and(y_true >= class_id, y_true < class_id + increment))[0]
        label = "{}-{}".format(str(class_id).rjust(2, "0"), str(class_id + increment - 1).rjust(2, "0"))
        all_acc[label] = np.around((y_pred[idxes] == y_true[idxes]).sum() * 100 / len(idxes), decimals=2)

    # Old accuracy
    idxes = np.where(y_true < nb_old)[0]
    all_acc["old"] = (
        0 if len(idxes) == 0 else np.around((y_pred[idxes] == y_true[idxes]).sum() * 100 / len(idxes), decimals=2)
    )

    # New accuracy
    idxes = np.where(y_true >= nb_old)[0]
    all_acc["new"] = np.around((y_pred[idxes] == y_true[idxes]).sum() * 100 / len(idxes), decimals=2)

    return all_acc


def split_images_labels(imgs):
    # split trainset.imgs in ImageFolder
    images = []
    labels = []
    for item in imgs:
        images.append(item[0])
        labels.append(item[1])

    return np.array(images), np.array(labels)


def accuracy_domain(y_pred, y_true, nb_old, increment=2, class_num=1):
    assert len(y_pred) == len(y_true), "Data length error."
    all_acc = {}
    all_acc["total"] = np.around((y_pred % class_num == y_true % class_num).sum() * 100 / len(y_true), decimals=2)

    # Grouped accuracy
    for class_id in range(0, np.max(y_true), increment):
        idxes = np.where(np.logical_and(y_true >= class_id, y_true < class_id + increment))[0]
        label = "{}-{}".format(str(class_id).rjust(2, "0"), str(class_id + increment - 1).rjust(2, "0"))
        all_acc[label] = np.around(
            ((y_pred[idxes] % class_num) == (y_true[idxes] % class_num)).sum() * 100 / len(idxes), decimals=2
        )

    # Old accuracy
    idxes = np.where(y_true < nb_old)[0]
    all_acc["old"] = (
        0
        if len(idxes) == 0
        else np.around(
            ((y_pred[idxes] % class_num) == (y_true[idxes] % class_num)).sum() * 100 / len(idxes), decimals=2
        )
    )

    # New accuracy
    idxes = np.where(y_true >= nb_old)[0]
    all_acc["new"] = np.around(
        ((y_pred[idxes] % class_num) == (y_true[idxes] % class_num)).sum() * 100 / len(idxes), decimals=2
    )

    return all_acc


def accuracy_domain_shot(
    y_pred,
    y_true,
    nb_old,
    increment=2,
    class_num=1,
    many_shot=None,
    medium_shot=None,
    few_shot=None,
):
    easy = False
    if class_num == 2:
        easy = True
    many_list = []
    medium_list = []
    few_list = []
    assert len(y_pred) == len(y_true), "Data length error."
    all_acc = {}
    all_acc["total"] = float(
        np.around((y_pred % class_num == y_true % class_num).sum() * 100 / len(y_true), decimals=2)
    )
    task_increment = class_num
    task = 0
    for class_id in range(0, np.max(y_true), task_increment):
        idxes = np.where(np.logical_and(y_true >= class_id, y_true < class_id + task_increment))[0]
        all_acc[f"{task}"] = float(
            np.around(
                ((y_pred[idxes] % class_num) == (y_true[idxes] % class_num)).sum()
                * 100
                / len(idxes),
                decimals=2,
            )
        )
        # Many-shot accuracy for this group
        if not easy:
            if many_shot is not None:
                many_idxes = np.where(np.isin(y_true[idxes], many_shot))[0]
                all_acc[f"{task}_many"] = float(
                    -1
                    if len(many_idxes) == 0
                    else np.around(
                        (
                            (y_pred[idxes][many_idxes] % class_num)
                            == (y_true[idxes][many_idxes] % class_num)
                        ).sum()
                        * 100
                        / len(many_idxes),
                        decimals=2,
                    )
                )
                if all_acc[f"{task}_many"] != -1:
                    many_list.append(all_acc[f"{task}_many"])
            # Medium-shot accuracy for this group
            if medium_shot is not None:
                medium_idxes = np.where(np.isin(y_true[idxes], medium_shot))[0]
                all_acc[f"{task}_medium"] = float(
                    -1
                    if len(medium_idxes) == 0
                    else np.around(
                        (
                            (y_pred[idxes][medium_idxes] % class_num)
                            == (y_true[idxes][medium_idxes] % class_num)
                        ).sum()
                        * 100
                        / len(medium_idxes),
                        decimals=2,
                    )
                )
                if all_acc[f"{task}_medium"] != -1:
                    medium_list.append(all_acc[f"{task}_medium"])
            # Few-shot accuracy for this group
            if few_shot is not None:
                few_idxes = np.where(
                    np.isin(y_true[idxes], few_shot),
                )[0]
                all_acc[f"{task}_few"] = float(
                    -1
                    if len(few_idxes) == 0
                    else np.around(
                        (
                            (y_pred[idxes][few_idxes] % class_num)
                            == (y_true[idxes][few_idxes] % class_num)
                        ).sum()
                        * 100
                        / len(few_idxes),
                        decimals=2,
                    )
                )
                if all_acc[f"{task}_few"] != -1:
                    few_list.append(all_acc[f"{task}_few"])
            print(f"{task}:many:{len(many_idxes)},medium:{len(medium_idxes)},few:{len(few_idxes)}")
        task += 1
    # Grouped accuracy with shot-based analysis

    if easy:
        domain_list = []
        for class_id in range(0, np.max(y_true), 1):
            idxes = np.where(np.logical_and(y_true >= class_id, y_true < class_id + increment))[0]
            domain_acc = float(
                np.around(
                    ((y_pred[idxes] % class_num) == (y_true[idxes] % class_num)).sum()
                    * 100
                    / len(idxes),
                    decimals=2,
                )
            )
            domain_list.append(domain_acc)
        all_acc["domain"] = np.mean(np.array(domain_list))
    else:
        for class_id in range(0, np.max(y_true), increment):
            idxes = np.where(np.logical_and(y_true >= class_id, y_true < class_id + increment))[0]
            label = "{}-{}".format(
                str(class_id).rjust(2, "0"), str(class_id + increment - 1).rjust(2, "0")
            )

            # Overall accuracy for this group
            all_acc[label] = float(
                np.around(
                    ((y_pred[idxes] % class_num) == (y_true[idxes] % class_num)).sum()
                    * 100
                    / len(idxes),
                    decimals=2,
                )
            )

    # Old accuracy
    idxes = np.where(y_true < nb_old)[0]
    all_acc["old"] = float(
        0
        if len(idxes) == 0
        else np.around(
            ((y_pred[idxes] % class_num) == (y_true[idxes] % class_num)).sum() * 100 / len(idxes),
            decimals=2,
        )
    )

    # New accuracy
    idxes = np.where(y_true >= nb_old)[0]
    all_acc["new"] = float(
        np.around(
            ((y_pred[idxes] % class_num) == (y_true[idxes] % class_num)).sum() * 100 / len(idxes),
            decimals=2,
        )
    )
    shot_acc = []
    if len(many_list) > 0:
        all_acc["many_shot"] = np.mean(many_list)
        shot_acc.append(all_acc["many_shot"])
    if len(medium_list) > 0:
        all_acc["medium_shot"] = np.mean(medium_list)
        shot_acc.append(all_acc["medium_shot"])
    if len(few_list) > 0:
        all_acc["few_shot"] = np.mean(few_list)
        shot_acc.append(all_acc["few_shot"])

    print("shot_acc:{}".format(shot_acc))
    return all_acc


def accuracy_binary(y_pred, y_true, nb_old, increment=2):
    assert len(y_pred) == len(y_true), "Data length error."
    all_acc = {}
    all_acc["total"] = np.around((y_pred % 2 == y_true % 2).sum() * 100 / len(y_true), decimals=2)

    # Grouped accuracy
    for class_id in range(0, np.max(y_true), increment):
        idxes = np.where(np.logical_and(y_true >= class_id, y_true < class_id + increment))[0]
        label = "{}-{}".format(str(class_id).rjust(2, "0"), str(class_id + increment - 1).rjust(2, "0"))
        all_acc[label] = np.around(((y_pred[idxes] % 2) == (y_true[idxes] % 2)).sum() * 100 / len(idxes), decimals=2)

    # Old accuracy
    idxes = np.where(y_true < nb_old)[0]
    # all_acc['old'] = 0 if len(idxes) == 0 else np.around((y_pred[idxes] == y_true[idxes]).sum()*100 / len(idxes),decimals=2)
    all_acc["old"] = (
        0
        if len(idxes) == 0
        else np.around(((y_pred[idxes] % 2) == (y_true[idxes] % 2)).sum() * 100 / len(idxes), decimals=2)
    )

    # New accuracy
    idxes = np.where(y_true >= nb_old)[0]
    all_acc["new"] = np.around(((y_pred[idxes] % 2) == (y_true[idxes] % 2)).sum() * 100 / len(idxes), decimals=2)

    return all_acc
