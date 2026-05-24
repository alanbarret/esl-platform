<?php
/**
 * Plugin Name: ESL Sign Language Video Generator
 * Plugin URI:  https://github.com/alanbarret/esl-platform
 * Description: Select any text on a news article and generate an Emirati Sign Language video instantly.
 * Version:     1.0.3
 * Author:      Alan Barrett
 * License:     MIT
 */

if (!defined('ABSPATH')) exit;

define('ESL_PLUGIN_VERSION', '1.0.3');
define('ESL_API_URL', get_option('esl_api_url', 'https://fluid-enforcement-teachers-mixer.trycloudflare.com'));

// ── Admin settings page ───────────────────────────────────────────────────────
add_action('admin_menu', function() {
    add_options_page('ESL Sign Plugin', 'ESL Sign Plugin', 'manage_options', 'esl-sign-plugin', 'esl_settings_page');
});

add_action('admin_init', function() {
    register_setting('esl_sign_plugin', 'esl_api_url');
    register_setting('esl_sign_plugin', 'esl_video_mode'); // skeleton or avatar
});

function esl_settings_page() { ?>
<div class="wrap">
  <h1>ESL Sign Language Plugin Settings</h1>
  <form method="post" action="options.php">
    <?php settings_fields('esl_sign_plugin'); ?>
    <table class="form-table">
      <tr>
        <th>API URL</th>
        <td>
          <input type="url" name="esl_api_url" value="<?php echo esc_attr(get_option('esl_api_url', 'https://fluid-enforcement-teachers-mixer.trycloudflare.com')); ?>" class="regular-text" />
          <p class="description">Your ESL Platform API base URL (without trailing slash)</p>
        </td>
      </tr>
      <tr>
        <th>Default Video Mode</th>
        <td>
          <select name="esl_video_mode">
            <option value="skeleton" <?php selected(get_option('esl_video_mode','skeleton'),'skeleton'); ?>>🦴 Skeleton</option>
            <option value="avatar"   <?php selected(get_option('esl_video_mode','skeleton'),'avatar'); ?>>👳 Arab Avatar</option>
          </select>
        </td>
      </tr>
    </table>
    <?php submit_button(); ?>
  </form>
  <hr>
  <h2>How to use</h2>
  <ol>
    <li>Visit any post or page on your site</li>
    <li>Select any text with your mouse</li>
    <li>Click the <strong>"Sign Language"</strong> button that appears</li>
    <li>The ESL video will generate and play in a popup</li>
  </ol>
</div>
<?php }

// u2500u2500 Hint bar on article pages u2500u2500u2500u2500u2500u2500u2500u2500u2500u2500u2500u2500u2500u2500u2500u2500u2500u2500u2500u2500u2500u2500u2500u2500u2500u2500u2500u2500u2500u2500
add_filter('the_content', function($content) {
    if (!is_singular()) return $content;
    $hint = '<div class="esl-article-hint">'
        . '<span class="esl-article-hint-icon">🤟</span>'
        . '<span><strong>Sign Language:</strong> Select any text and tap the purple button to see it in Emirates Sign Language.</span>'
        . '</div>';
    return $hint . $content;
});

// ── Enqueue frontend assets ───────────────────────────────────────────────────
add_action('wp_enqueue_scripts', function() {
    // Load on all front-end pages so the floating trigger button works everywhere
    // (text selection-to-sign is useful on home, archives, single posts, etc.)
    if (is_admin()) return;

    // Use filemtime() for cache-busting so any CSS/JS edit forces a refresh
    $css_path = plugin_dir_path(__FILE__) . 'assets/esl-sign.css';
    $art_path = plugin_dir_path(__FILE__) . 'assets/article-style.css';
    $js_path  = plugin_dir_path(__FILE__) . 'assets/esl-sign.js';
    $css_ver = file_exists($css_path) ? (string) filemtime($css_path) : ESL_PLUGIN_VERSION;
    $art_ver = file_exists($art_path) ? (string) filemtime($art_path) : ESL_PLUGIN_VERSION;
    $js_ver  = file_exists($js_path)  ? (string) filemtime($js_path)  : ESL_PLUGIN_VERSION;

    wp_enqueue_style(
        'esl-sign-plugin',
        plugin_dir_url(__FILE__) . 'assets/esl-sign.css',
        [], $css_ver
    );
    wp_enqueue_style(
        'esl-article-style',
        plugin_dir_url(__FILE__) . 'assets/article-style.css',
        [], $art_ver
    );
    wp_enqueue_script(
        'esl-sign-plugin',
        plugin_dir_url(__FILE__) . 'assets/esl-sign.js',
        [], $js_ver, true
    );
    wp_localize_script('esl-sign-plugin', 'eslConfig', [
        'apiUrl'    => esc_url(get_option('esl_api_url', ESL_API_URL)),
        'videoMode' => get_option('esl_video_mode', 'skeleton'),
        'nonce'     => wp_create_nonce('esl_sign_nonce'),
        'ajaxUrl'   => admin_url('admin-ajax.php'),
    ]);
});

// ── AJAX proxy (avoids CORS issues on some servers) ──────────────────────────
add_action('wp_ajax_esl_translate',        'esl_ajax_translate');
add_action('wp_ajax_nopriv_esl_translate', 'esl_ajax_translate');

function esl_ajax_translate() {
    check_ajax_referer('esl_sign_nonce', 'nonce');

    $text = sanitize_textarea_field($_POST['text'] ?? '');
    if (empty($text)) wp_send_json_error('No text provided');

    $api_url = get_option('esl_api_url', ESL_API_URL);
    $response = wp_remote_post("$api_url/api/v1/translate", [
        'timeout' => 30,
        'headers' => ['Content-Type' => 'application/json'],
        'body'    => wp_json_encode(['text' => $text]),
    ]);

    if (is_wp_error($response)) {
        wp_send_json_error($response->get_error_message());
    }

    $body = json_decode(wp_remote_retrieve_body($response), true);
    wp_send_json_success([
        'tokens'          => $body['gloss_tokens'] ?? [],
        'video_url'       => $body['video_url']       ?? null,
        'avatar_video_url'=> $body['avatar_video_url'] ?? null,
        'api_base'        => $api_url,
    ]);
}
