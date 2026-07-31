import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os

def create_attrition_flowchart():
    # ==========================================================
    # DIRECTORY SETUP
    # ==========================================================
    save_dir = '/Users/edwardyao/Documents/PURM/gender_project_output/'
    os.makedirs(save_dir, exist_ok=True)
    out_path = os.path.join(save_dir, 'poster_flowchart.png')

    # ==========================================================
    # ATTRITION DATA & CALCULATIONS 
    # ==========================================================
    START_N = 6467 
    
    # Text blocked out to have equal-width lines to kill horizontal whitespace
    exclusions = [
        ("Excluded: EEG not\noutpatient routine\n<= 4 hours", 1840),
        ("Excluded: no LLM-\nconfirmed diagnosis\n(n=668) or unknown\nsubtype (n=324)", 992),
        ("Excluded: no\ndocumented seizure\nfrequency", 708),
        ("Excluded: other\ngender (n=1), missing\nage (n=3), < 18 (n=7)", 11)
    ]
    
    main_labels = [
        "All patients\nwith EEG data",
        "Outpatient routine\nEEG <= 4 hours",
        "LLM-confirmed\ndiagnosis & known\nepilepsy subtype",
        "Documented seizure\nfrequency\n(primary cohort)",
        "Final study cohort\n(valid adult\ndemographics)"
    ]
    
    ns = [START_N]
    for _, drop in exclusions:
        ns.append(ns[-1] - drop)

    # ==========================================================
    # FIGURE CONFIGURATION & STYLING (Strict Original Box Sizes)
    # ==========================================================
    fig, ax = plt.subplots(figsize=(5, 8.5))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    
    x_main = 0.26
    x_exc = 0.74
    
    # YOUR EXACT ORIGINAL DIMENSIONS
    box_w = 0.45  
    box_h = 0.125  
    
    y_start = 0.93
    y_end = 0.07  
    n_steps = len(main_labels)
    y_step = (y_start - y_end) / (n_steps - 1)

    # Title
    ax.text(x_main, y_start + (box_h / 2) + 0.02, "Study participant flow", 
            fontsize=13, fontweight='bold', ha='center', va='bottom')

    def draw_box(center_x, center_y, text, facecolor, edgecolor):
        x0 = center_x - box_w / 2
        y0 = center_y - box_h / 2
        rect = patches.Rectangle((x0, y0), box_w, box_h, facecolor=facecolor, 
                                 edgecolor=edgecolor, linewidth=1.5, zorder=2)
        ax.add_patch(rect)
        
        # FIX: Pushed font size way up and compressed line spacing to fill the whitespace
        ax.text(center_x, center_y, text, ha='center', va='center', 
                fontsize=12, linespacing=0.95, zorder=3)

    # ==========================================================
    # DRAWING LOOP
    # ==========================================================
    for i in range(n_steps):
        y = y_start - i * y_step
        
        fc_main = "#d0ddec" if i < 3 else "#cbe3d6"
        ec_main = "#497eb9" if i < 3 else "#2c8a5c"
        
        draw_box(x_main, y, f"{main_labels[i]}\nN = {ns[i]}", fc_main, ec_main)
        
        if i < len(exclusions):
            y_next = y_start - (i + 1) * y_step
            y_mid = (y + y_next) / 2
            
            # Solid vertical line
            ax.plot([x_main, x_main], [y - box_h/2, y_next + box_h/2], 
                    color='#606060', linewidth=2, zorder=1)
            
            # Vertical Arrowhead
            ax.annotate("", xy=(x_main, y_next + box_h / 2), xytext=(x_main, y_next + box_h / 2 + 0.01),
                        arrowprops=dict(arrowstyle="-|>", color='#606060', 
                                        lw=2, mutation_scale=15), zorder=1)
            
            exc_text, exc_drop = exclusions[i]
            draw_box(x_exc, y_mid, f"{exc_text}\nN = {exc_drop}", "#f4d1c6", "#d9531e")
            
            # Horizontal red arrow
            ax.annotate("", xy=(x_exc - box_w / 2, y_mid), xytext=(x_main, y_mid),
                        arrowprops=dict(arrowstyle="-|>", color='#d9531e', 
                                        lw=2, mutation_scale=15), zorder=1)

    # ==========================================================
    # RENDER & SAVE
    # ==========================================================
    plt.tight_layout(pad=0)
    plt.savefig(out_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"Flowchart successfully generated and saved to:\n -> {out_path}")

if __name__ == "__main__":
    create_attrition_flowchart()