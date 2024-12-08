document.getElementById('upload-form').addEventListener('submit', async (event) => {
    event.preventDefault();
    const formData = new FormData(event.target);
    const statusDiv = document.getElementById('status');
    const progressContainer = document.getElementById('progress-container');
    const progressBar = document.getElementById('progress-bar');
    const progressText = document.getElementById('progress-text'); // % এর জন্য
    const outputSection = document.getElementById('output-section');
    const videoElement = document.getElementById('output-video');
    const downloadLink = document.getElementById('download-link');

    // Reset UI
    statusDiv.textContent = "Uploading and processing...";
    progressContainer.style.display = 'block';
    progressBar.style.width = '0%';
    progressText.textContent = "Progress: 0%"; // শুরুতে 0%
    outputSection.style.display = 'none';

    // Upload video
    const response = await fetch('/upload', {
        method: 'POST',
        body: formData
    });

    const result = await response.json();
    if (result.error) {
        statusDiv.textContent = `Error: ${result.error}`;
        progressContainer.style.display = 'none';
        return;
    }

    const taskId = result.task_id;
    const outputUrl = result.output_url;

    // Track progress
    const progressInterval = setInterval(async () => {
        const progressResponse = await fetch(`/progress/${taskId}`);
        const progressData = await progressResponse.json();

        if (progressData.error) {
            clearInterval(progressInterval);
            statusDiv.textContent = "Error fetching progress.";
            return;
        }

        const progressPercentage = progressData.progress;
        progressBar.style.width = `${progressPercentage}%`;
        progressText.textContent = `Progress: ${progressPercentage}%`; // % দেখানো

        if (progressPercentage >= 100) {
            clearInterval(progressInterval);
            statusDiv.textContent = "Processing complete!";
            progressContainer.style.display = 'none';

            // Show processed video and download link
            videoElement.src = outputUrl;
            videoElement.style.display = 'block';
            downloadLink.href = outputUrl;
            downloadLink.style.display = 'block';
            outputSection.style.display = 'block';
        }
    }, 1000);
});
