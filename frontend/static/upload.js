// Function to render the statistics and table
function renderStats(data) {
    // Show the results section
    document.getElementById('results-section').style.display = 'block';

    // Update the simple stats
    document.getElementById('timeSaved').textContent = data.total_time_saved.toFixed(2);
    document.getElementById('timeSavedPercentage').textContent = data.time_saved_percentage;
    document.getElementById('movesRequired').textContent = data.moves_required;
    document.getElementById('totalSkus').textContent = data.total_skus;
    document.getElementById('zonesUsed').textContent = data.zones_used;
    document.getElementById('totalPicks').textContent = data.total_picks;

    // Update the zone distribution
    const zoneDist = data.zone_distribution;
    document.getElementById('zoneA').textContent = zoneDist.A;
    document.getElementById('zoneB').textContent = zoneDist.B;
    document.getElementById('zoneC').textContent = zoneDist.C;

    // Populate the top time savers table
    const tableBody = document.querySelector('#top-savers-table tbody');
    tableBody.innerHTML = ''; // Clear existing rows
    data.top_time_savers.forEach(sku => {
        const row = document.createElement('tr');
        row.innerHTML = `
            <td>${sku.sku}</td>
            <td>${sku.picks}</td>
            <td>${sku.before_location}</td>
            <td>${sku.after_location}</td>
            <td>${sku.time_saved_sec.toFixed(2)}</td>
        `;
        tableBody.appendChild(row);
    });
}

// ... Inside your uploadForm.addEventListener('submit', async (event) => { ...
// After the slotting response is received and parsed
const slottingResult = await slottingResponse.json();

console.log('Slotting analysis result:', slottingResult);

// Render the images
if (slottingResult.before_heatmap_url && slottingResult.after_heatmap_url) {
    beforeHeatmap.src = slottingResult.before_heatmap_url;
    afterHeatmap.src = slottingResult.after_heatmap_url;
    uploadStatus.textContent = 'Slotting analysis complete! Heatmaps rendered.';
    uploadStatus.style.color = 'green';
} else {
    throw new Error('Heatmap URLs are missing from the server response.');
}

// Render the statistics
// The JSON response has nested data, so you need to access the correct keys
// for example, optimization_summary and detailed_summary
const optimizationMetrics = slottingResult.optimization_metrics;
const detailedSummary = slottingResult.detailed_summary;

// Combine the relevant data points to pass to your rendering function
const dataToRender = {
    total_time_saved: optimizationMetrics.total_time_saved,
    time_saved_percentage: optimizationMetrics.time_saved_percentage,
    moves_required: optimizationMetrics.moves_required,
    total_skus: optimizationMetrics.total_skus,
    zones_used: detailedSummary.zones_used,
    total_picks: detailedSummary.total_picks,
    zone_distribution: detailedSummary.zone_distribution,
    top_time_savers: detailedSummary.top_time_savers
};

renderStats(dataToRender);