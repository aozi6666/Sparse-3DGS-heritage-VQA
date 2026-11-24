import csv
import json
import statistics

def plot_results(out, rows, previous):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    past=list(map(json.loads,previous.read_text().splitlines()))
    records=[r for r in past if r["budget"] in (0,2,4,8)]+rows
    assert len(records)==len({(r["id"],r["budget"]) for r in records})==7140
    budgets=[0,2,4,8,16,20,24,28,32]
    table=[]
    for duration in ("short","medium","long","overall"):
        data={}
        for k in budgets+["pure_aks"]:
            group=[r for r in records if r["budget"]==k and (duration=="overall" or r["duration"]==duration)]
            n=len(group)
            data[k]=100*sum(r["correct"] for r in group)/n
            table.append(dict(duration=duration,arm=k,questions=n,correct=sum(r["correct"] for r in group),
                              accuracy_pct=data[k],mean_complete_frames=statistics.mean(len(r["frames"]) for r in group)))
        if duration=="overall":
            continue
        fig,ax=plt.subplots(figsize=(7,4.2),layout="constrained")
        ax.plot(budgets,[data[k] for k in budgets],"o-",color="#205990",label="Codec + AKS replacement")
        ax.axhline(data["pure_aks"],color="#a14e2d",linestyle="--",label="AKS full frames only")
        ax.set(xlabel="Replacement cap K",ylabel="QA accuracy (%)",title=f"VideoMME | {duration.capitalize()} videos (n={n} questions)",xticks=budgets)
        ax.grid(axis="y",alpha=.22)
        ax.legend()
        for fmt in ["pdf","png"]:
            fig.savefig(out/f"budget_{duration}.{fmt}",dpi=200)
        plt.close(fig)
    with (out/"all_budgets.csv").open("w",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=list(table[0]))
        writer.writeheader()
        writer.writerows(table)
