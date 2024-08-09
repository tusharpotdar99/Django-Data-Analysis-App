// Ensure that the DOM is fully loaded before running the script
document.addEventListener('DOMContentLoaded', function() {
    // Select the plot image and container
    const plotImage = document.querySelector('.plot-img');
    const zoomContainer = document.querySelector('.zoom-container');

    if (plotImage && zoomContainer) {
        // Set initial zoom level
        let zoomLevel = 1;

        // Add event listeners for zoom in and zoom out
        document.getElementById('zoomIn').addEventListener('click', function() {
            zoomLevel += 0.1;
            plotImage.style.transform = `scale(${zoomLevel})`;
        });

        document.getElementById('zoomOut').addEventListener('click', function() {
            zoomLevel = Math.max(1, zoomLevel - 0.1); // Prevent zooming out too much
            plotImage.style.transform = `scale(${zoomLevel})`;
        });
    }
});
