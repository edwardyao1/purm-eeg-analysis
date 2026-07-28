import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os

def create_attrition_flowchart():
    # ==========================================================
    # DIRECTORY SETUP
    # ==========================================================
    save_dir = '/Users/edwardyao/Documents/PURM/gender_project_output/'
    os.makedirs(save_dir, exist_ok=True)
    out_path = os.path.join(save_dir, 'two_cohort_attrition_flowchart.png')

    # ==========================================================
    # ATTRITION DATA & CALCULATIONS
    # ==========================================================
    START_N = 6467 
    
    exclusions = [
        ("Excluded: EEG not outpatient\nroutine <= 4 hours", 1840),
        ("Excluded: no LLM-confirmed\nepilepsy diagnosis", 668),
        ("Excluded: no documented\nseizure frequency", 708),
        ("Excluded: gender not\nMale or Female", 1),
        ("Excluded: subtype not\nGeneralized or Focal", 324),
        ("Excluded: missing age", 3),
        ("Excluded: under age 18", 7)
    ]
    
    main_labels = [
        "All patients with EEG data",
        "Outpatient routine EEG <= 4 hours",
        "LLM-confirmed epilepsy diagnosis",
        "Documented seizure frequency\n(primary cohort)",
        "Valid Gender (Male or Female)",
        "Known epilepsy subtype\n(Generalized or Focal)",
        "Valid age recorded",
        "Final study cohort"
    ]
    
    ns = [START_N]
    for _, drop in exclusions:
        ns.append(ns[-1] - drop)

    # ==========================================================
    # FIGURE CONFIGURATION & STYLING
    # ==========================================================
    fig, ax = plt.subplots(figsize=(10, 16))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    
    # Layout Coordinates
    x_main = 0.28
    x_exc = 0.74
    
    # Decreased width and height to reduce empty space
    box_w = 0.38  
    box_h = 0.055  
    
    y_start = 0.90
    y_end = 0.05
    n_steps = len(main_labels)
    y_step = (y_start - y_end) / (n_steps - 1)

    # Title explicitly centered on top of the first blue box
    ax.text(x_main, y_start + (box_h / 2) + 0.02, "Study participant flow", 
            fontsize=16, fontweight='bold', ha='center', va='bottom')

    def draw_box(center_x, center_y, text, facecolor, edgecolor):
        """Draws a strictly sized rectangle patch to ensure uniform boxes."""
        x0 = center_x - box_w / 2
        y0 = center_y - box_h / 2
        rect = patches.Rectangle((x0, y0), box_w, box_h, facecolor=facecolor, 
                                 edgecolor=edgecolor, linewidth=1.5, zorder=2)
        ax.add_patch(rect)
        # Increased font size to make the text take up more space within the box
        ax.text(center_x, center_y, text, ha='center', va='center', fontsize=12.5, zorder=3)

    # ==========================================================
    # DRAWING LOOP
    # ==========================================================
    for i in range(n_steps):
        y = y_start - i * y_step
        
        # Color coding: First 3 steps are blue, the rest are green
        fc_main = "#d0ddec" if i < 3 else "#cbe3d6"
        ec_main = "#497eb9" if i < 3 else "#2c8a5c"
        
        # Draw Main Inclusion Box
        draw_box(x_main, y, f"{main_labels[i]}\nN = {ns[i]}", fc_main, ec_main)
        
        # Draw Arrows and Exclusion Boxes (if not the last step)
        if i < len(exclusions):
            y_next = y_start - (i + 1) * y_step
            y_mid = (y + y_next) / 2
            
            # 1. Vertical Down Arrow (Main Flow)
            # Ends exactly at the top boundary of the next box
            ax.annotate("", xy=(x_main, y_next + box_h / 2), xytext=(x_main, y - box_h / 2),
                        arrowprops=dict(facecolor='#606060', edgecolor='#606060', 
                                        width=2, headwidth=10, headlength=10, shrink=0),
                        zorder=1)
            
            # 2. Draw Branching Exclusion Box
            exc_text, exc_drop = exclusions[i]
            draw_box(x_exc, y_mid, f"{exc_text}\nN = {exc_drop}", "#f4d1c6", "#d9531e")
            
            # 3. Horizontal Branching Arrow
            # Connects directly from the vertical flow line to the absolute left edge of the exclusion box
            ax.annotate("", xy=(x_exc - box_w / 2, y_mid), xytext=(x_main, y_mid),
                        arrowprops=dict(facecolor='#d9531e', edgecolor='#d9531e', 
                                        width=2, headwidth=10, headlength=10, shrink=0),
                        zorder=1)

    # ==========================================================
    # RENDER & SAVE
    # ==========================================================
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"Flowchart successfully generated and saved to:\n -> {out_path}")

if __name__ == "__main__":
    create_attrition_flowchart()