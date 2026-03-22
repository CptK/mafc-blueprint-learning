/**
 * Sets up the resizable divider between main graph area and sidebar.
 * @param {HTMLElement} dividerEl
 * @param {HTMLElement} appEl
 * @param {() => any} getCy - returns the current cytoscape instance (or null)
 */
export function setupDividerDrag(dividerEl, appEl, getCy) {
  let isDraggingDivider = false;

  function stopDividerDrag() {
    if (!isDraggingDivider) return;
    isDraggingDivider = false;
    dividerEl.classList.remove("dragging");
    document.body.style.userSelect = "";
  }

  dividerEl.addEventListener("pointerdown", (event) => {
    if (window.innerWidth <= 1100) return;
    isDraggingDivider = true;
    dividerEl.classList.add("dragging");
    dividerEl.setPointerCapture(event.pointerId);
    document.body.style.userSelect = "none";
  });

  dividerEl.addEventListener("pointermove", (event) => {
    if (!isDraggingDivider || window.innerWidth <= 1100) return;
    const bounds = appEl.getBoundingClientRect();
    const minMainWidth = 360;
    const minSidebarWidth = 280;
    const dividerWidth = 12;
    const maxSidebarWidth = Math.max(minSidebarWidth, bounds.width - minMainWidth - dividerWidth);
    const rawSidebarWidth = bounds.right - event.clientX;
    const sidebarWidth = Math.min(maxSidebarWidth, Math.max(minSidebarWidth, rawSidebarWidth));
    appEl.style.gridTemplateColumns = `minmax(${minMainWidth}px, 1fr) ${dividerWidth}px minmax(${minSidebarWidth}px, ${sidebarWidth}px)`;
    const cy = getCy();
    if (cy) {
      cy.resize();
      cy.fit(undefined, 50);
    }
  });

  dividerEl.addEventListener("pointerup", stopDividerDrag);
  dividerEl.addEventListener("pointercancel", stopDividerDrag);
  window.addEventListener("pointerup", stopDividerDrag);
}
