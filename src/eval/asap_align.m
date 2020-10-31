function asap_align(path, offset, length)
  template_points = dlmread('../data/smpl/templates/template_points.txt', ' ');
  for i = offset:(offset+length-1)
    if mod(i, 100) == 0
      corres_file = strcat([path, '/', int2str(i), '.corres']);
      data = dlmread(corres_file, ' ');
      sampled_points = data(:, 1:3);
      gt_points = data(:, 4:6);
      sampled_corres = data(:, 7);
      target_points = template_points(sampled_corres+1, :);
      f12 = flow_prediction(target_points', sampled_points', 16, 1)';
      Idx = knnsearch(target_points, template_points);
      deformed_points = target_points + f12;
      deform_diffs = deformed_points - sampled_points;
      %mean(vecnorm(deform_diffs', 2))
      flow = f12(Idx(:, 1), :);
      deformed_points = template_points + flow;
      Idx2 = knnsearch(deformed_points, sampled_points);
      pred_points = template_points(Idx2(:, 1), :);
      diff = pred_points - gt_points;
      errors = vecnorm(diff');
      diff_before = target_points - gt_points;
      errors_before = (vecnorm(diff_before'));
      errors_pac = [errors_before; errors]';
      errors_file = strcat([path, '/', int2str(i), '_errors.mat']);
      save(errors_file, 'errors_pac', 'Idx2');
      %mean(errors_before)
      %mean(errors)
    end
  end
end
